import pytest
from mock import patch, MagicMock, mock_open
from sys import version_info

from botocore.exceptions import ClientError
from botocore.stub import Stubber

import boto3
from moto import mock_ecr, mock_iam, mock_sts

from bentoml.deployment.sagemaker import (
    _parse_aws_client_exception_or_raise,
    _cleanup_sagemaker_model,
    _cleanup_sagemaker_endpoint_config,
    get_arn_role_from_current_aws_user,
    SageMakerDeploymentOperator,
)
from bentoml.proto.deployment_pb2 import Deployment, DeploymentSpec
from bentoml.proto.repository_pb2 import Bento, BentoServiceMetadata, GetBentoResponse
from bentoml.proto.status_pb2 import Status
from tests.deployment.sagemaker.sagemaker_moto import moto_mock_sagemaker


@pytest.fixture(scope='function')
def sagemaker_client():
    return boto3.client('sagemaker', 'us-west-2')


def test_sagemaker_handle_client_errors(sagemaker_client):
    sagemaker_client = boto3.client('sagemaker', 'us-west-2')
    stubber = Stubber(sagemaker_client)

    stubber.add_client_error(
        method='create_endpoint', service_error_code='ValidationException'
    )
    stubber.activate()
    result = None

    with pytest.raises(ClientError) as error:
        sagemaker_client.create_endpoint(
            EndpointName='Test', EndpointConfigName='test-config'
        )
    result = _parse_aws_client_exception_or_raise(error.value)

    assert result.status_code == Status.NOT_FOUND

    stubber.add_client_error('describe_endpoint', 'InvalidSignatureException')
    stubber.activate()
    result = None
    with pytest.raises(ClientError) as e:
        sagemaker_client.describe_endpoint(EndpointName='Test')
    result = _parse_aws_client_exception_or_raise(e.value)
    assert result.status_code == Status.UNAUTHENTICATED


def test_cleanup_sagemaker_model(sagemaker_client):
    sagemaker_client = boto3.client('sagemaker', 'us-west-2')
    stubber = Stubber(sagemaker_client)
    stubber.add_client_error(
        method='delete_model', service_error_code='ValidationException'
    )
    stubber.activate()

    error_status = _cleanup_sagemaker_model(
        sagemaker_client, 'test_name', 'test_version'
    )
    assert error_status.status_code == Status.NOT_FOUND

    stubber.add_client_error(
        method='delete_model', service_error_code='InvalidSignatureException'
    )
    error_status = _cleanup_sagemaker_model(
        sagemaker_client, 'test_name', 'test_version'
    )
    assert error_status.status_code == Status.UNAUTHENTICATED

    stubber.add_client_error(
        method='delete_model',
        service_error_code='RandomError',
        service_message='random',
    )
    with pytest.raises(ClientError) as error:
        _cleanup_sagemaker_model(sagemaker_client, 'test_name', 'test_version')
    assert error.value.operation_name == 'DeleteModel'
    assert error.value.response['Error']['Code'] == 'RandomError'


def test_cleanup_sagemaker_endpoint_config(sagemaker_client):
    stubber = Stubber(sagemaker_client)
    stubber.add_client_error(
        method='delete_endpoint_config', service_error_code='ValidationException'
    )
    stubber.activate()

    error_status = _cleanup_sagemaker_endpoint_config(
        sagemaker_client, 'test_name', 'test_version'
    )
    assert error_status.status_code == Status.NOT_FOUND

    stubber.add_client_error(
        method='delete_endpoint_config', service_error_code='InvalidSignatureException'
    )
    error_status = _cleanup_sagemaker_endpoint_config(
        sagemaker_client, 'test_name', 'test_version'
    )
    assert error_status.status_code == Status.UNAUTHENTICATED

    stubber.add_client_error(
        method='delete_endpoint_config',
        service_error_code='RandomError',
        service_message='random',
    )
    with pytest.raises(ClientError) as error:
        _cleanup_sagemaker_endpoint_config(
            sagemaker_client, 'test_name', 'test_version'
        )
    assert error.value.operation_name == 'DeleteEndpointConfig'
    assert error.value.response['Error']['Code'] == 'RandomError'


ROLE_PATH_ARN_RESULT = 'arn:aws:us-west-2:999'
USER_PATH_ARN_RESULT = 'arn:aws:us-west-2:888'


def test_get_arn_from_aws_user():
    def mock_role_path_call(self, operation_name, kwarg):
        if operation_name == 'GetCallerIdentity':
            return {'Arn': 'something:something:role/random'}
        elif operation_name == 'GetRole':
            return {'Role': {'Arn': ROLE_PATH_ARN_RESULT}}
        else:
            raise Exception(
                'This test does not handle operation: {}'.format(operation_name)
            )

    @patch('botocore.client.BaseClient._make_api_call', new=mock_role_path_call)
    def role_path():
        return get_arn_role_from_current_aws_user()

    assert role_path() == ROLE_PATH_ARN_RESULT

    def mock_user_path_call(self, operation_name, kwarg):
        if operation_name == 'GetCallerIdentity':
            return {'Arn': 'something:something:user/random'}
        elif operation_name == 'ListRoles':
            return {
                "Roles": [
                    {
                        "AssumeRolePolicyDocument": {
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                                }
                            ]
                        },
                        "Arn": USER_PATH_ARN_RESULT,
                    }
                ]
            }
        else:
            raise Exception(
                'This test does not handle operation: {}'.format(operation_name)
            )

    @patch('botocore.client.BaseClient._make_api_call', new=mock_user_path_call)
    def user_path():
        return get_arn_role_from_current_aws_user()

    assert user_path() == USER_PATH_ARN_RESULT


if version_info.major >= 3:
    mock_open_param_value = 'builtins.open'
else:
    mock_open_param_value = '__builtin__.open'


TEST_AWS_REGION = 'us-west-2'
TEST_DEPLOYMENT_NAME = 'my_deployment'
TEST_DEPLOYMENT_NAMESPACE = 'my_company'
TEST_DEPLOYMENT_BENTO_NAME = 'mybento'
TEST_DEPLOYMENT_BENTO_VERSION = 'v1.1.0'
TEST_BENTO_API_NAME = 'predict'
TEST_DEPLOYMENT_INSTANCE_COUNT = 1
TEST_DEPLOYMENT_INSTANCE_TYPE = 'm3-xlarge'
TEST_DEPLOYMENT_BENTO_LOCAL_URI = '/fake/path/bento/bundle'


def mock_aws_services_for_sagemaker(func):
    @mock_ecr
    @mock_iam
    @mock_sts
    @moto_mock_sagemaker
    def mock_wrapper(*args, **kwargs):
        ecr_client = boto3.client('ecr', region_name=TEST_AWS_REGION)
        repo_name = TEST_DEPLOYMENT_BENTO_NAME + '-sagemaker'
        ecr_client.create_repository(repositoryName=repo_name)

        iam_role_policy = """
        {
            "Version": "2019-10-10",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "*",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    }
                }
            ]
        }
        """
        iam_client = boto3.client('iam', region_name=TEST_AWS_REGION)
        iam_client.create_role(
            RoleName="moto", AssumeRolePolicyDocument=iam_role_policy
        )

        return func(*args, **kwargs)

    return mock_wrapper


def mock_sagemaker_deployment_wrapper(func):
    @mock_aws_services_for_sagemaker
    @patch('subprocess.check_output', autospec=True)
    @patch('docker.APIClient.build', autospec=True)
    @patch('docker.APIClient.push', autospec=True)
    @patch(mock_open_param_value, mock_open(read_data='test'), create=True)
    @patch('shutil.copytree', autospec=True)
    @patch('os.chmod', autospec=True)
    def mock_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return mock_wrapper


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply():
    def mock_get_bento(is_local=True):
        bento_pb = Bento(
            name=TEST_DEPLOYMENT_BENTO_NAME, version=TEST_DEPLOYMENT_BENTO_VERSION
        )
        # BentoUri.StorageType.LOCAL
        if is_local:
            bento_pb.uri.type = 1
        bento_pb.uri.uri = TEST_DEPLOYMENT_BENTO_LOCAL_URI
        api = BentoServiceMetadata.BentoServiceApi(name=TEST_BENTO_API_NAME)
        bento_pb.bento_service_metadata.apis.extend([api])
        return GetBentoResponse(bento=bento_pb)

    test_deployment_pb = Deployment(
        name=TEST_DEPLOYMENT_NAME,
        namespace=TEST_DEPLOYMENT_NAMESPACE,
        spec=DeploymentSpec(
            bento_name=TEST_DEPLOYMENT_BENTO_NAME,
            bento_version=TEST_DEPLOYMENT_BENTO_VERSION,
        ),
    )
    test_deployment_pb.spec.sagemaker_operator_config.api_name = TEST_BENTO_API_NAME
    test_deployment_pb.spec.sagemaker_operator_config.region = TEST_AWS_REGION
    test_deployment_pb.spec.sagemaker_operator_config.instance_count = (
        TEST_DEPLOYMENT_INSTANCE_COUNT
    )
    test_deployment_pb.spec.sagemaker_operator_config.instance_type = (
        TEST_DEPLOYMENT_INSTANCE_TYPE
    )
    # DeploymentSpec.DeploymentOperator.AWS_SAGEMAKER
    test_deployment_pb.spec.operator = 2

    deployment_operator = SageMakerDeploymentOperator()
    fake_yatai_service = MagicMock()
    fake_yatai_service.GetBento = lambda uri: mock_get_bento(is_local=False)
    result_pb = deployment_operator.apply(test_deployment_pb, fake_yatai_service)
    assert result_pb.status.status_code == Status.INTERNAL
    assert result_pb.status.error_message.startswith(
        'BentoML currently only support local repository'
    )

    fake_yatai_service.GetBento = lambda uri: mock_get_bento()
    result_pb = deployment_operator.apply(test_deployment_pb, fake_yatai_service)
    assert result_pb.status.status_code == Status.OK
    assert result_pb.deployment.name == TEST_DEPLOYMENT_NAME
