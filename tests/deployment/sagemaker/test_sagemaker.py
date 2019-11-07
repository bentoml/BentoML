from __future__ import absolute_import

import botocore
import pytest
from mock import patch, MagicMock, mock_open
from sys import version_info

from botocore.exceptions import ClientError
from botocore.stub import Stubber

import boto3
from moto import mock_ecr, mock_iam, mock_sts

from bentoml.deployment.sagemaker import (
    _parse_aws_client_exception_or_raise,
    get_arn_role_from_current_aws_user,
    SageMakerDeploymentOperator,
)
from bentoml.proto.deployment_pb2 import Deployment, DeploymentSpec
from bentoml.proto.repository_pb2 import Bento, BentoServiceMetadata, GetBentoResponse, \
    BentoUri
from bentoml.proto.status_pb2 import Status
from tests.deployment.sagemaker.sagemaker_moto import moto_mock_sagemaker


def test_sagemaker_handle_client_errors():
    error = ClientError(
        error_response={
            'Error': {'Code': 'ValidationException', 'Message': 'error message'}
        },
        operation_name='failed_operation',
    )
    result = _parse_aws_client_exception_or_raise(error)
    assert result.status_code == Status.NOT_FOUND

    invalid_error = ClientError(
        error_response={
            'Error': {'Code': 'InvalidSignatureException', 'Message': 'error message'}
        },
        operation_name='DescribeEndpoint',
    )
    result = _parse_aws_client_exception_or_raise(invalid_error)
    assert result.status_code == Status.UNAUTHENTICATED


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

    with patch('botocore.client.BaseClient._make_api_call', new=mock_user_path_call):
        assert get_arn_role_from_current_aws_user() == USER_PATH_ARN_RESULT


TEST_AWS_REGION = 'us-west-2'
TEST_DEPLOYMENT_NAME = 'my_deployment'
TEST_DEPLOYMENT_NAMESPACE = 'my_company'
TEST_DEPLOYMENT_BENTO_NAME = 'my_bento'
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
        ecr_client = boto3.client('ecr', TEST_AWS_REGION)
        repo_name = TEST_DEPLOYMENT_BENTO_NAME + '-sagemaker'
        try:
            ecr_client.create_repository(repositoryName=repo_name)
        except ecr_client.exceptions.RepositoryAlreadyExistsException:
            print('Repository already exists')

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
        iam_client = boto3.client('iam', TEST_AWS_REGION)
        iam_client.create_role(
            RoleName="moto", AssumeRolePolicyDocument=iam_role_policy
        )
        return func(*args, **kwargs)

    return mock_wrapper


def mock_sagemaker_deployment_wrapper(func):
    if version_info.major >= 3:
        mock_open_param_value = 'builtins.open'
    else:
        mock_open_param_value = '__builtin__.open'

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


def create_yatai_service_mock(repo_storage_type=BentoUri.LOCAL):
    def mock_get_bento(repo_storage_type):
        bento_pb = Bento(
            name=TEST_DEPLOYMENT_BENTO_NAME, version=TEST_DEPLOYMENT_BENTO_VERSION
        )
        if repo_storage_type == BentoUri.LOCAL:
            bento_pb.uri.uri = TEST_DEPLOYMENT_BENTO_LOCAL_URI
        bento_pb.uri.type = repo_storage_type
        api = BentoServiceMetadata.BentoServiceApi(name=TEST_BENTO_API_NAME)
        bento_pb.bento_service_metadata.apis.extend([api])
        return GetBentoResponse(bento=bento_pb)

    fake_yatai_service = MagicMock()
    fake_yatai_service.GetBento = lambda pb_request: mock_get_bento(repo_storage_type)

    return fake_yatai_service


def generate_sagemaker_deployment_pb():
    test_deployment_pb = Deployment(
        name=TEST_DEPLOYMENT_NAME, namespace=TEST_DEPLOYMENT_NAMESPACE
    )
    test_deployment_pb.spec.bento_name = TEST_DEPLOYMENT_BENTO_NAME
    test_deployment_pb.spec.bento_version = TEST_DEPLOYMENT_BENTO_VERSION
    test_deployment_pb.spec.sagemaker_operator_config.api_name = TEST_BENTO_API_NAME
    test_deployment_pb.spec.sagemaker_operator_config.region = TEST_AWS_REGION
    test_deployment_pb.spec.sagemaker_operator_config.instance_count = (
        TEST_DEPLOYMENT_INSTANCE_COUNT
    )
    test_deployment_pb.spec.sagemaker_operator_config.instance_type = (
        TEST_DEPLOYMENT_INSTANCE_TYPE
    )
    test_deployment_pb.spec.operator = DeploymentSpec.AWS_SAGEMAKER

    return test_deployment_pb


def raise_(ex):
    raise ex


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_fail_not_local_repo(
    mock_chmod, mock_copytree, mock_docker_push, mock_docker_build, mock_check_output
):
    yatai_service = create_yatai_service_mock(repo_storage_type=BentoUri.S3)
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator()
    result_pb = deployment_operator.apply(sagemaker_deployment_pb, yatai_service)
    assert result_pb.status.status_code == Status.INTERNAL
    assert result_pb.status.error_message.startswith(
        'BentoML currently only support local repository'
    )


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_success(
    mock_chmod, mock_copytree, mock_docker_push, mock_docker_build, mock_check_output
):
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator()
    result_pb = deployment_operator.apply(sagemaker_deployment_pb, yatai_service)
    assert result_pb.status.status_code == Status.OK
    assert result_pb.deployment.name == TEST_DEPLOYMENT_NAME


# @mock_sagemaker_deployment_wrapper
# def test_sagemaker_apply_model_already_exists(
#     mock_chmod, mock_copytree, mock_docker_push, mock_docker_build, mock_check_output
# ):
#     yatai_service = create_yatai_service_mock()
#     sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
#     deployment_operator = SageMakerDeploymentOperator()
#     with pytest.raises(ValueError) as error:
#         result_pb = deployment_operator.apply(sagemaker_deployment_pb, yatai_service)
#     print(error.value)
#     assert False


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_create_model_fail(
    mock_chmod, mock_copytree, mock_docker_push, mock_docker_build, mock_check_output
):
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator()

    orig = botocore.client.BaseClient._make_api_call

    def fail_create_model_random(self, operation_name, kwarg):
        if operation_name == 'CreateModel':
            raise ClientError({'Error': {'Code': 'Random'}}, 'CreateModel')
        else:
            return orig(self, operation_name, kwarg)

    with pytest.raises(ClientError) as error:
        with patch(
            'botocore.client.BaseClient._make_api_call', new=fail_create_model_random
        ):
            deployment_operator.apply(sagemaker_deployment_pb, yatai_service)
    assert error.value.operation_name == 'CreateModel'
    assert error.value.response['Error']['Code'] == 'Random'

    def fail_create_model_validation(self, operation_name, kwarg):
        if operation_name == 'CreateModel':
            raise ClientError(
                {'Error': {'Code': 'ValidationException', 'Message': 'failed message'}},
                'CreateModel',
            )
        else:
            return orig(self, operation_name, kwarg)

    with patch(
        'botocore.client.BaseClient._make_api_call', new=fail_create_model_validation
    ):
        result = deployment_operator.apply(sagemaker_deployment_pb, yatai_service)
    assert result.status.status_code == Status.NOT_FOUND
    assert result.status.error_message.startswith(
        'Failed to create model for SageMaker'
    )


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_delete_model_fail(
    mock_chmod, mock_copytree, mock_docker_push, mock_docker_build, mock_check_output
):
    orig = botocore.client.BaseClient._make_api_call
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator()

    def fail_delete_model(self, operation_name, kwarg):
        if operation_name == 'DeleteModel':
            raise ClientError(
                {'Error': {'Code': 'ValidationException', 'Message': 'failed message'}},
                'DeleteModel',
            )
        elif operation_name == 'CreateEndpoint':
            raise ClientError({}, 'CreateEndpoint')
        else:
            return orig(self, operation_name, kwarg)

    with patch('botocore.client.BaseClient._make_api_call', new=fail_delete_model):
        result = deployment_operator.apply(sagemaker_deployment_pb, yatai_service)
    assert result.status.status_code == Status.NOT_FOUND
    assert result.status.error_message.startswith(
        'Failed to clean up model after unsuccessfully'
    )


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_duplicate_endpoint(
    mock_chmod, mock_copytree, mock_docker_push, mock_docker_build, mock_check_output
):
    orig = botocore.client.BaseClient._make_api_call
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator()
    deployment_operator.apply(sagemaker_deployment_pb, yatai_service)

    endpoint_name = '{ns}-{name}'.format(
        ns=TEST_DEPLOYMENT_NAMESPACE, name=TEST_DEPLOYMENT_BENTO_NAME
    )
    expect_value = 'Endpoint {} already exists'.format(endpoint_name.replace('_', '-'))

    def mock_ok_return(self, op_name, kwargs):
        if op_name == 'CreateModel' or op_name == 'CreateEndpointConfig':
            return ''
        else:
            return orig(self, op_name, kwargs)

    with patch('botocore.client.BaseClient._make_api_call', new=mock_ok_return):
        with pytest.raises(ValueError) as error:
            deployment_operator.apply(sagemaker_deployment_pb, yatai_service)
    assert str(error.value) == expect_value
