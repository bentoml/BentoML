import uuid

import mock
import pytest
from click.testing import CliRunner

from bentoml import load
from bentoml.cli import create_bento_service_cli, create_bentoml_cli
from bentoml.exceptions import InvalidArgument
from bentoml.yatai.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    Deployment,
    DeploymentSpec,
)
from bentoml.yatai.status import Status


def test_service_labels_on_save_and_load(bento_service):
    with pytest.raises(InvalidArgument):
        bento_service.save(labels='test_string:abc')
    with pytest.raises(InvalidArgument):
        bento_service.save(labels={'key_containers_&': 'value'})

    saved_dir = bento_service.save(
        labels={'example_key': 'values_can_contains_and-and.s'}
    )

    loaded_service = load(saved_dir)
    assert loaded_service.labels['example_key'] == 'values_can_contains_and-and.s'


def test_service_labels_on_cli_info(bento_service):
    saved_path = bento_service.save(labels={'key': 'value1', 'key2': 'value2'})

    runner = CliRunner()
    cli = create_bento_service_cli()
    result = runner.invoke(cli.commands['info'], [saved_path])
    assert result.exit_code == 0
    output = result.output.strip()
    assert '"labels"' in output
    assert '"key2": "value2"' in output


def test_label_selectors_on_cli_list(bento_service):
    runner = CliRunner()
    cli = create_bentoml_cli()
    failed_result = runner.invoke(
        cli.commands['list'], ['--labels', '"incorrect label query"']
    )
    assert failed_result.exit_code == 1
    assert type(failed_result.exception) == AssertionError
    assert str(failed_result.exception) == 'Operator "label" is invalid'

    unique_label_value = uuid.uuid4().hex
    bento_service.save(labels={'test_id': unique_label_value})

    success_result = runner.invoke(
        cli.commands['list'], ['--labels', f'test_id in ({unique_label_value})']
    )
    assert success_result.exit_code == 0
    assert f'{bento_service.name}:{bento_service.version}' in success_result.output


def test_label_selectors_on_cli_get(bento_service):
    runner = CliRunner()
    cli = create_bentoml_cli()

    unique_label_value = uuid.uuid4().hex
    bento_service.save(labels={'test_id': unique_label_value})

    success_result = runner.invoke(
        cli.commands['get'],
        [bento_service.name, '--labels', f'test_id in ({unique_label_value})'],
    )
    assert success_result.exit_code == 0
    assert f'{bento_service.name}:{bento_service.version}' in success_result.output


def test_deployment_labels(bento_service):
    runner = CliRunner()
    cli = create_bentoml_cli()

    failed_result = runner.invoke(
        cli.commands['lambda'],
        [
            'deploy',
            'failed-name',
            '-b',
            'ExampleBentoService:version',
            '--labels',
            'test=abc',
        ],
    )
    assert failed_result.exit_code == 2

    with mock.patch(
        'bentoml.yatai.deployment.aws_lambda.operator.AwsLambdaDeploymentOperator.add'
    ) as mock_operator_add:
        bento_name = 'MockService'
        bento_version = 'MockVersion'
        deployment_name = f'test-label-{uuid.uuid4().hex[:8]}'
        deployment_namespace = 'test-namespace'
        mocked_deployment_pb = Deployment(
            name=deployment_name, namespace=deployment_namespace
        )
        mocked_deployment_pb.spec.bento_name = bento_name
        mocked_deployment_pb.spec.bento_version = bento_version
        mocked_deployment_pb.spec.operator = DeploymentSpec.AWS_LAMBDA
        mocked_deployment_pb.spec.aws_lambda_operator_config.memory_size = 1000
        mocked_deployment_pb.spec.aws_lambda_operator_config.timeout = 60
        mocked_deployment_pb.spec.aws_lambda_operator_config.region = 'us-west-2'
        mock_operator_add.return_value = ApplyDeploymentResponse(
            status=Status.OK(), deployment=mocked_deployment_pb
        )

        success_result = runner.invoke(
            cli.commands['lambda'],
            [
                'deploy',
                deployment_name,
                '-b',
                f'{bento_name}:{bento_version}',
                '--namespace',
                deployment_namespace,
                '--labels',
                'created_by:admin,cicd:passed',
                '--region',
                'us-west-2',
            ],
        )
        assert success_result.exit_code == 0

        list_result = runner.invoke(
            cli.commands['deployment'],
            [
                'list',
                '--labels',
                'created_by=admin,cicd NotIn (failed, unsuccessful)',
                '--output',
                'wide',
            ],
        )
        print(list_result.output)
        assert list_result.exit_code == 0
        assert deployment_name in list_result.output.strip()
        assert 'created_by:admin' in list_result.output.strip()
