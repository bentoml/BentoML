import os
from sys import version_info

from mock import MagicMock, patch, Mock, mock_open
from moto import mock_cloudformation
from ruamel.yaml import YAML

from bentoml.deployment.serverless.aws_lambda import (
    generate_aws_lambda_serverless_config,
    generate_aws_lambda_handler_py,
    AwsLambdaDeploymentOperator,
)
from bentoml.proto.deployment_pb2 import Deployment
from bentoml.proto.repository_pb2 import (
    Bento,
    BentoServiceMetadata,
    GetBentoResponse,
    BentoUri,
)
from bentoml.proto.status_pb2 import Status
from bentoml.deployment.serverless import serverless_utils


def test_generate_aws_lambda_serverless_config(tmpdir):
    python_version = '3.7.2'
    deployment_name = 'test_deployment_lambda'
    api_names = ['predict']
    region = 'us-west-test'
    namespace = 'namespace'
    generate_aws_lambda_serverless_config(
        python_version, deployment_name, api_names, str(tmpdir), region, namespace
    )
    config_path = os.path.join(str(tmpdir), 'serverless.yml')
    yaml = YAML()
    with open(config_path, 'rb') as f:
        yaml_data = yaml.load(f.read())
    assert yaml_data['service'] == deployment_name
    assert yaml_data['functions']['predict']['handler'] == 'handler.predict'
    assert yaml_data['provider']['region'] == 'us-west-test'


def test_generate_aws_lambda_handler_py(tmpdir):
    bento_name = 'bento_name'
    api_names = ['predict', 'second_predict']
    generate_aws_lambda_handler_py(bento_name, api_names, str(tmpdir))

    def fake_predict(value):
        return value

    import sys

    class fake_bento(object):
        def get_service_api(self, name):
            if name == 'predict':
                mock_api = Mock()
                mock_api.handle_aws_lambda_event = fake_predict
                return mock_api

    mock_module = Mock()
    mock_module.load = Mock(return_value=fake_bento())
    sys.modules['bento_name'] = mock_module

    sys.path.insert(1, str(tmpdir))
    from handler import predict

    assert predict.__module__ == 'handler'
    assert predict(1, None) == 1
    sys.path.remove(str(tmpdir))


def mock_describe_cf(self, name):
    if name == 'DescribeStacks':
        return {
            "Stacks": [
                {
                    "Outputs": [
                        {
                            "OutputKey": "ServiceEndpoint",
                            "OutputValue": 'https://base_url.lambda.aws.com',
                        }
                    ]
                }
            ]
        }
    return


def mock_aws_lambda_deployment_wrapper(func):
    if version_info.major >= 3:
        mock_open_param_value = 'builtins.open'
    else:
        mock_open_param_value = '__builtin__.open'

    @patch('subprocess.check_output', autospec=True)
    @patch('subprocess.check_call', autospec=True)
    @patch.object(serverless_utils, 'check_nodejs_compatible_version', autospec=True)
    @patch(mock_open_param_value, mock_open(), create=True)
    @patch('shutil.copytree', autospec=True)
    @patch('shutil.copy', autospec=True)
    @patch.object(serverless_utils, 'init_serverless_project_dir', autospec=True)
    @patch('subprocess.Popen')
    @patch('botocore.client.BaseClient._make_api_call', new=mock_describe_cf)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def create_yatai_service_mock(repo_storage_type=BentoUri.LOCAL):
    bento_pb = Bento(name='bento_test_name', version='version1.1.1')
    if repo_storage_type == BentoUri.LOCAL:
        bento_pb.uri.uri = '/fake/path/to/bundle'
    bento_pb.uri.type = repo_storage_type
    api = BentoServiceMetadata.BentoServiceApi(name='predict')
    bento_pb.bento_service_metadata.apis.extend([api])
    get_bento_response = GetBentoResponse(bento=bento_pb)

    yatai_service_mock = MagicMock()
    yatai_service_mock.GetBento.return_value = get_bento_response
    return yatai_service_mock


def generate_lambda_deployment_pb():
    test_deployment_pb = Deployment(name='test_aws_lambda', namespace='test-namespace')
    test_deployment_pb.spec.bento_name = 'bento_name'
    test_deployment_pb.spec.bento_version = 'v1.0.0'
    # DeploymentSpec.DeploymentOperator.AWS_LAMBDA
    test_deployment_pb.spec.operator = 3
    test_deployment_pb.spec.aws_lambda_operator_config.region = 'us-west-2'
    test_deployment_pb.spec.aws_lambda_operator_config.api_name = 'predict'

    return test_deployment_pb


@mock_aws_lambda_deployment_wrapper
def test_aws_lambda_apply_failed_only_local_repo(
    mock_popen,
    mock_init_serverless,
    mock_copy,
    mock_copytree,
    mock_check_nodejs,
    mock_checkcall,
    mock_checkoutput,
):
    test_deployment_pb = generate_lambda_deployment_pb()
    yatai_service_mock = create_yatai_service_mock(BentoUri.UNSET)
    deployment_operator = AwsLambdaDeploymentOperator()
    result_pb = deployment_operator.apply(test_deployment_pb, yatai_service_mock)
    assert result_pb.status.status_code == Status.INTERNAL
    assert result_pb.status.error_message.startswith(
        'BentoML currently only support local repository'
    )


@mock_aws_lambda_deployment_wrapper
def test_aws_lambda_apply_success(
    mock_popen,
    mock_init_serverless,
    mock_copy,
    mock_copytree,
    mock_check_nodejs,
    mock_checkcall,
    mock_checkoutput,
):
    test_deployment_pb = generate_lambda_deployment_pb()
    yatai_service_mock = create_yatai_service_mock()
    deployment_operator = AwsLambdaDeploymentOperator()
    result_pb = deployment_operator.apply(test_deployment_pb, yatai_service_mock)
    assert result_pb.status.status_code == Status.OK
    assert result_pb.deployment.name == 'test_aws_lambda'


@mock_cloudformation
def test_aws_lambda_describe_failed_no_formation():
    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_lambda_deployment_pb()
    deployment_operator = AwsLambdaDeploymentOperator()
    result_pb = deployment_operator.describe(test_deployment_pb, yatai_service_mock)
    assert result_pb.status.status_code == Status.INTERNAL
    assert result_pb.status.error_message.startswith(
        'An error occurred (ValidationError)'
    )


def test_aws_lambda_describe_success():
    def mock_cf_response(self, op_name, kwarg):
        if op_name == 'DescribeStacks':
            return {
                'Stacks': [
                    {
                        'Outputs': [
                            {
                                'OutputValue': 'https://fake.aws.amazon.com/',
                                'OutputKey': 'ServiceEndpoint',
                            }
                        ]
                    }
                ]
            }
        else:
            raise Exception('This test does not handle operation {}'.format(op_name))

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_lambda_deployment_pb()
    with patch('botocore.client.BaseClient._make_api_call', new=mock_cf_response):
        deployment_operator = AwsLambdaDeploymentOperator()
        result_pb = deployment_operator.describe(test_deployment_pb, yatai_service_mock)
        assert result_pb.status.status_code == Status.OK
