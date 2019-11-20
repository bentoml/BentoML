import os

import boto3
from mock import MagicMock, patch, Mock
from moto import mock_s3
from ruamel.yaml import YAML

from bentoml.deployment.aws_lambda import (
    AwsLambdaDeploymentOperator,
    generate_aws_lambda_template_config,
)
from bentoml.deployment.aws_lambda.utils import generate_aws_lambda_app_py
from bentoml.proto.deployment_pb2 import Deployment, DeploymentState
from bentoml.proto.repository_pb2 import (
    Bento,
    BentoUri,
    BentoServiceMetadata,
    GetBentoResponse,
)
from bentoml.proto.status_pb2 import Status


fake_s3_bucket_name = 'fake_deployment_bucket'
fake_s3_prefix = 'prefix'
fake_s3_path = 's3://{}/{}'.format(fake_s3_bucket_name, fake_s3_prefix)


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
    test_deployment_pb.spec.aws_lambda_operator_config.s3_path = fake_s3_path
    test_deployment_pb.spec.aws_lambda_operator_config.s3_region = 'us-west-2'
    test_deployment_pb.spec.aws_lambda_operator_config.memory_size = 3008
    test_deployment_pb.spec.aws_lambda_operator_config.timeout = 6

    return test_deployment_pb


def test_generate_aws_lambda_app_py(tmpdir):
    bento_name = 'bento_name'
    api_names = ['predict', 'second_predict']
    generate_aws_lambda_app_py(
        str(tmpdir),
        s3_bucket=fake_s3_bucket_name,
        artifacts_prefix='fake_artifact_prefix',
        bento_name=bento_name,
        api_names=api_names,
    )

    def fake_predict(value):
        return value

    class fake_bento(object):
        def _load_artifacts(self, path):
            return

        def get_service_api(self, name):
            if name == 'predict':
                mock_api = Mock()
                mock_api.handle_aws_lambda_event = fake_predict
                return mock_api

    import sys

    sys.path.insert(1, str(tmpdir))

    def mock_lambda_app(func):
        @mock_s3
        @patch(
            'bentoml.deployment.aws_lambda.utils.download_artifacts_for_lambda_function'
        )
        @patch('bentoml.bundler.load_bento_service_class', return_value=fake_bento)
        def mock_wrapper(*args, **kwargs):
            conn = boto3.client('s3', region_name='us-west-2')
            conn.create_bucket(Bucket=fake_s3_bucket_name)
            fake_artifact_key = 'fake_artifact_prefix/model.pkl'
            conn.put_object(
                Bucket=fake_s3_bucket_name, Key=fake_artifact_key, Body='fakebody'
            )
            return func(*args, **kwargs)

        return mock_wrapper

    @mock_lambda_app
    def return_predict_func(mock_load_service, mock_download_artifacts):
        from app import predict

        return predict

    predict = return_predict_func()
    assert predict.__module__ == 'app'
    assert predict(1, None) == 1
    sys.path.remove(str(tmpdir))


def test_generate_aws_lambda_template_yaml(tmpdir):
    deployment_name = 'deployment_name'
    api_names = ['predict', 'classify']
    s3_bucket_name = 'fake_bucket'
    py_runtime = 'python3.7'
    memory_size = 3008
    timeout = 6
    generate_aws_lambda_template_config(
        str(tmpdir),
        deployment_name,
        api_names,
        s3_bucket_name,
        py_runtime,
        memory_size,
        timeout,
    )
    template_path = os.path.join(str(tmpdir), 'template.yaml')
    yaml = YAML()
    with open(template_path, 'rb') as f:
        yaml_data = yaml.load(f.read())
    assert yaml_data['Resources']['predict']['Properties']['Runtime'] == py_runtime
    assert yaml_data['Resources']['classify']['Properties']['Handler'] == 'app.classify'


def mock_lambda_related_operations(func):
    @patch('subprocess.check_output', autospec=True)
    @mock_s3
    def mock_wrapper(*args, **kwargs):
        conn = boto3.client('s3', region_name='us-west-2')
        conn.create_bucket(Bucket=fake_s3_bucket_name)
        return func(*args, **kwargs)

    return mock_wrapper()


@mock_lambda_related_operations
def test_aws_lambda_apply_fails_no_artifacts_directory(mock_checkoutput):
    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_lambda_deployment_pb()
    deployment_operator = AwsLambdaDeploymentOperator()
    result_pb = deployment_operator.apply(test_deployment_pb, yatai_service_mock, None)
    assert result_pb.status.status_code == Status.INTERNAL


def mock_cf(self, op_name, kwargs):
    if op_name == 'DescribeStacks':
        return {}
    else:
        raise Exception('Operation {} is not mocked in this test'.format(op_name))


# @patch('botocore.client.BaseClient._make_api_call', new=mock_cf)
# @mock_lambda_related_operations
# @patch('shutil.rmtree', autospec=True)
# @patch('shutil.copytree', autospec=True)
# @patch('shutil.copy', autospec=True)
# @patch('os.listdir')
# @patch('subprocess.Popen')
# def test_aws_lambda_apply(
#     mock_popen, mock_checkoutput, mock_listdir, mock_copy, mock_copytree, mock_rmtree
# ):
#     yatai_service_mock = create_yatai_service_mock()
#     test_deployment_pb = generate_lambda_deployment_pb()
#     deployment_operator = AwsLambdaDeploymentOperator()
#     result_pb = deployment_operator.apply(test_deployment_pb, yatai_service_mock, None)
#     print(result_pb)
#     assert False


def test_aws_lambda_describe_still_in_progress():
    def mock_cf_response(self, op_name, kwarg):
        if op_name == 'DescribeStacks':
            return {'Stacks': [{'StackStatus': 'CREATE_IN_PROGRESS'}]}
        else:
            raise Exception('This test does not handle operation {}'.format(op_name))

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_lambda_deployment_pb()
    with patch('botocore.client.BaseClient._make_api_call', new=mock_cf_response):
        deployment_operator = AwsLambdaDeploymentOperator()
        result_pb = deployment_operator.describe(test_deployment_pb, yatai_service_mock)
        assert result_pb.status.status_code == Status.OK
        assert result_pb.state.state == DeploymentState.PENDING


def test_aws_lambda_describe_success():
    def mock_cf_response(self, op_name, kwarg):
        if op_name == 'DescribeStacks':
            return {
                'Stacks': [
                    {
                        'StackStatus': 'CREATE_COMPLETE',
                        'Outputs': [
                            {
                                'OutputKey': 'EndpointUrl',
                                'OutputValue': 'https://somefakelink.amazonaws.com'
                                '/prod/predict',
                            }
                        ],
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
        assert result_pb.state.state == DeploymentState.RUNNING
