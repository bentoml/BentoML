import botocore
import pytest
from mock import patch, MagicMock

from botocore.exceptions import ClientError

import boto3
from moto import mock_ecr, mock_iam, mock_sts

from bentoml.yatai.deployment.sagemaker.operator import (
    _aws_client_error_to_bentoml_exception,
    get_arn_role_from_current_aws_user,
)
from bentoml.yatai.deployment.sagemaker.operator import SageMakerDeploymentOperator
from bentoml.yatai.proto.deployment_pb2 import Deployment, DeploymentSpec
from bentoml.yatai.proto.repository_pb2 import (
    Bento,
    BentoServiceMetadata,
    GetBentoResponse,
    BentoUri,
)
from bentoml.yatai.proto.status_pb2 import Status
from bentoml.exceptions import AWSServiceError
from tests.deployment.sagemaker.sagemaker_moto import moto_mock_sagemaker


def test_sagemaker_handle_client_errors():
    error = ClientError(
        error_response={
            'Error': {'Code': 'ValidationException', 'Message': 'error message'}
        },
        operation_name='failed_operation',
    )
    exception = _aws_client_error_to_bentoml_exception(error)
    assert isinstance(exception, AWSServiceError)


ROLE_PATH_ARN_RESULT = 'arn:aws:us-west-2:999'
USER_PATH_ARN_RESULT = 'arn:aws:us-west-2:888'


def test_get_arn_from_aws_user():
    def mock_role_path_call(
        self, operation_name, kwarg
    ):  # pylint: disable=unused-argument
        if operation_name == 'GetCallerIdentity':
            return {'Arn': 'something:something:role/random'}
        elif operation_name == 'GetRole':
            return {'Role': {'Arn': ROLE_PATH_ARN_RESULT}}
        else:
            raise Exception(
                'This test does not handle operation: {}'.format(operation_name)
            )

    with patch('botocore.client.BaseClient._make_api_call', new=mock_role_path_call):
        assert get_arn_role_from_current_aws_user() == ROLE_PATH_ARN_RESULT

    def mock_user_path_call(
        self, operation_name, kwarg
    ):  # pylint: disable=unused-argument
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
    @mock_aws_services_for_sagemaker
    @patch('subprocess.check_output', MagicMock())
    @patch('docker.APIClient.build', MagicMock())
    @patch('docker.APIClient.push', MagicMock())
    @patch(
        'bentoml.yatai.deployment.sagemaker.operator._init_sagemaker_project',
        MagicMock(),
    )
    @patch(
        'bentoml.yatai.deployment.sagemaker.operator.get_default_aws_region',
        MagicMock(return_value='mock_region'),
    )
    def mock_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return mock_wrapper


def create_yatai_service_mock(repo_storage_type=BentoUri.LOCAL):
    bento_pb = Bento(
        name=TEST_DEPLOYMENT_BENTO_NAME, version=TEST_DEPLOYMENT_BENTO_VERSION
    )
    if repo_storage_type == BentoUri.LOCAL:
        bento_pb.uri.uri = TEST_DEPLOYMENT_BENTO_LOCAL_URI
    bento_pb.uri.type = repo_storage_type
    api = BentoServiceMetadata.BentoServiceApi(name=TEST_BENTO_API_NAME)
    bento_pb.bento_service_metadata.apis.extend([api])
    get_bento_response = GetBentoResponse(bento=bento_pb)

    yatai_service_mock = MagicMock()
    yatai_service_mock.GetBento.return_value = get_bento_response

    return yatai_service_mock


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
def test_sagemaker_apply_fail_not_local_repo():
    yatai_service = create_yatai_service_mock(repo_storage_type=BentoUri.UNSET)
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator(yatai_service)
    result_pb = deployment_operator.add(sagemaker_deployment_pb)
    assert result_pb.status.status_code == Status.INTERNAL
    assert result_pb.status.error_message.startswith('BentoML currently not support')


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_success():
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator(yatai_service)
    result_pb = deployment_operator.add(sagemaker_deployment_pb)
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
def test_sagemaker_apply_create_model_fail():
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator(yatai_service)

    orig = botocore.client.BaseClient._make_api_call

    def fail_create_model_random(self, operation_name, kwarg):
        if operation_name == 'CreateModel':
            raise ClientError({'Error': {'Code': 'Random'}}, 'CreateModel')
        else:
            return orig(self, operation_name, kwarg)

    with patch(
        'botocore.client.BaseClient._make_api_call', new=fail_create_model_random
    ):
        failed_result = deployment_operator.add(sagemaker_deployment_pb)
    assert failed_result.status.status_code == Status.INTERNAL
    assert failed_result.status.error_message.startswith(
        'Failed to create sagemaker model'
    )

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
        result = deployment_operator.add(sagemaker_deployment_pb)
    assert result.status.status_code == Status.INTERNAL
    assert result.status.error_message.startswith('Failed to create sagemaker model')


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_delete_model_fail():
    orig = botocore.client.BaseClient._make_api_call
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator(yatai_service)

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
        result = deployment_operator.add(sagemaker_deployment_pb)
    assert result.status.status_code == Status.INTERNAL
    assert result.status.error_message.startswith('Failed to cleanup sagemaker model')


@mock_sagemaker_deployment_wrapper
def test_sagemaker_apply_duplicate_endpoint():
    orig = botocore.client.BaseClient._make_api_call
    yatai_service = create_yatai_service_mock()
    sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator(yatai_service)
    deployment_operator.add(sagemaker_deployment_pb)

    endpoint_name = '{ns}-{name}'.format(
        ns=TEST_DEPLOYMENT_NAMESPACE, name=TEST_DEPLOYMENT_NAME
    )
    expect_value = 'Endpoint {} already exists'.format(endpoint_name.replace('_', '-'))

    def mock_ok_return(self, op_name, kwargs):
        if op_name == 'CreateModel' or op_name == 'CreateEndpointConfig':
            return ''
        else:
            return orig(self, op_name, kwargs)

    with patch('botocore.client.BaseClient._make_api_call', new=mock_ok_return):
        with pytest.raises(ValueError) as error:
            deployment_operator.add(sagemaker_deployment_pb)
    assert str(error.value) == expect_value


@mock_sagemaker_deployment_wrapper
def test_sagemaker_update_deployment_with_new_bento_service_tag():
    mocked_yatai_service = create_yatai_service_mock()
    mocked_sagemaker_deployment_pb = generate_sagemaker_deployment_pb()
    deployment_operator = SageMakerDeploymentOperator(mocked_yatai_service)
    deployment_operator.add(mocked_sagemaker_deployment_pb)
    mocked_sagemaker_deployment_pb_with_new_bento_service_tag = (
        generate_sagemaker_deployment_pb()
    )
    mocked_sagemaker_deployment_pb_with_new_bento_service_tag.spec.bento_version = (
        'NEW_BENTO_VERSION'
    )
    update_sagemaker_deployment_result = deployment_operator.update(
        mocked_sagemaker_deployment_pb_with_new_bento_service_tag,
        mocked_sagemaker_deployment_pb,
    )
    assert update_sagemaker_deployment_result.status.status_code == Status.OK
