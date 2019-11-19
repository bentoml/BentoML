import os

from mock import MagicMock, patch, Mock, mock_open
from moto import mock_cloudformation
from ruamel.yaml import YAML

from bentoml.deployment.aws_lambda import AwsLambdaDeploymentOperator
from bentoml.deployment.aws_lambda.utils import generate_aws_lambda_app_py
from bentoml.proto.deployment_pb2 import Deployment, DeploymentState
from bentoml.proto.repository_pb2 import (
    Bento,
    BentoUri,
    BentoServiceMetadata,
    GetBentoResponse,
)
from bentoml.proto.status_pb2 import Status


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
    test_deployment_pb.spec.aws_lambda_operator_config.s3_path = 's3://fake_bucket/path'

    return test_deployment_pb


def test_generate_aws_lambda_app_py(tmpdir):
    bento_name = 'bento_name'
    api_names = ['predict', 'second_predict']
    generate_aws_lambda_app_py(
        str(tmpdir),
        s3_bucket='fake_bucket',
        artifacts_prefix='fake_artifacts_prefix',
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

    def mock_aws_calls(self, op_name, kwargs):
        if op_name == 'ListObjects':
            return {'Contents': [{'Key': 'some/s3/key'}]}
        elif op_name == 'DownloadFile':
            return True
        else:
            raise Exception(
                'Operation {} does not support in this mock'.format(op_name)
            )

    os = MagicMock()
    os.path.isfile.return_value = True
    # os.path.join.return_value = 'fake/path'
    os.mkdir = MagicMock()

    import six
    if six.PY3:
        mock_open_param_value = 'builtins.open'
    else:
        mock_open_param_value = '__builtin__.open'

    @patch('os.path.isfile', os.path.isfile)
    #@patch('os.path.join', os.path.join)
    @patch(mock_open_param_value, mock_open(), create=True)
    @patch('os.mkdir', os.mkdir)
    @patch('botocore.client.BaseClient._make_api_call', new=mock_aws_calls)
    @patch('bentoml.bundler.load_bento_service_class', return_value=fake_bento)
    def return_predict_func(mock_load_class):
        mock_load_class.return_value = fake_bento
        from app import predict

        return predict

    predict = return_predict_func()
    assert predict.__module__ == 'app'
    assert predict(1, None) == 1
    sys.path.remove(str(tmpdir))


def test_generate_aws_lambda_template_yaml():
    assert True


def test_aws_lambda_apply():
    assert True


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
                                'OutputValue': 'https://somefakelink.amazonaws.com/prod/predict',
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
