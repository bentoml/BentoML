from mock import patch, MagicMock, mock_open
from sys import version_info

from botocore.exceptions import ClientError
from botocore.stub import Stubber

import botocore
import boto3

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


def test_sagemaker_handle_client_errors():
    client = boto3.client('sagemaker', 'us-west-2')
    stubber = Stubber(client)

    stubber.add_client_error(
        method='create_endpoint', service_error_code='ValidationException'
    )
    stubber.activate()
    result = None
    try:
        client.create_endpoint(EndpointName='Test', EndpointConfigName='test-config')
    except ClientError as e:
        result = _parse_aws_client_exception_or_raise(e)

    assert result.status_code == Status.NOT_FOUND

    stubber.add_client_error('describe_endpoint', 'InvalidSignatureException')
    stubber.activate()
    result = None
    try:
        client.describe_endpoint(EndpointName='Test')
    except ClientError as e:
        result = _parse_aws_client_exception_or_raise(e)
    assert result.status_code == Status.UNAUTHENTICATED


def test_cleanup_sagemaker_functions():
    client = boto3.client('sagemaker', 'us-west-2')
    stubber = Stubber(client)
    stubber.add_client_error(
        method='delete_model', service_error_code='ValidationException'
    )
    stubber.activate()

    error_status = _cleanup_sagemaker_model(client, 'test_name', 'test_version')
    assert error_status.status_code == Status.NOT_FOUND

    stubber.add_response('delete_endpoint_config', {})
    _cleanup_sagemaker_endpoint_config(client, 'test-name', 'test-version')


orig = botocore.client.BaseClient._make_api_call


def mock_aws_api_calls(self, operation_name, kwarg):
    if operation_name == 'GetCallerIdentity':
        return {'Arn': 'something:something:role/random'}
    elif operation_name == 'GetRole':
        return {'Role': {'Arn': 'arn:aws:us-west-2:999888'}}
    elif operation_name == 'GetAuthorizationToken':
        return {
            'authorizationData': [
                {
                    # b64encoded string of 'user:password'
                    'authorizationToken': 'dXNlcjpwYXNzd29yZA==',
                    'proxyEndpoint': 'https://fake.regsitry.aws.com',
                }
            ]
        }
    elif operation_name == 'DescribeRepositories':
        return
    elif operation_name == 'CreateRepository':
        return
    elif operation_name == 'CreateModel':
        return {}
    elif operation_name == 'CreateEndpointConfig':
        return {}
    elif operation_name == 'CreateEndpoint':
        return {}
    elif operation_name == 'UpdateEndpoint':
        return {}
    return orig(self, operation_name, kwarg)


@patch('botocore.client.BaseClient._make_api_call', new=mock_aws_api_calls)
def test_get_arn_from_aws_user():
    arn_role = get_arn_role_from_current_aws_user()
    assert arn_role == 'arn:aws:us-west-2:999888'


def mock_get_bento():
    bento_pb = Bento(name='bento_test_name', version='version1.1.1')
    # BentoUri.StorageType.LOCAL
    bento_pb.uri.type = 1
    bento_pb.uri.uri = '/fake/path/to/bundle'
    api = BentoServiceMetadata.BentoServiceApi(name='predict')
    bento_pb.bento_service_metadata.apis.extend([api])
    return GetBentoResponse(bento=bento_pb)


if version_info.major >= 3:
    mock_open_param_value = 'builtins.open'
else:
    mock_open_param_value = '__builtin__.open'


@patch('subprocess.check_output', autospec=True)
@patch('docker.APIClient.build', autospec=True)
@patch('docker.APIClient.push', autospec=True)
@patch('botocore.client.BaseClient._make_api_call', new=mock_aws_api_calls)
@patch(mock_open_param_value, mock_open(read_data='test'), create=True)
@patch('shutil.copytree', autospec=True)
@patch('os.chmod', autospec=True)
@patch(
    'bentoml.deployment.sagemaker.create_push_docker_image_to_ecr',
    new=lambda x, y, z: 'https://fake.aws.com',
)
def test_sagemaker_apply(
    mock_chmod, mock_copytree, mock_docker_push, mock_docker_build, mock_check_output
):
    test_deployment_pb = Deployment(
        name='test',
        namespace='test-namespace',
        spec=DeploymentSpec(bento_name='bento_test_name', bento_version='version1.1.1'),
    )
    test_deployment_pb.spec.sagemaker_operator_config.api_name = 'predict'
    test_deployment_pb.spec.sagemaker_operator_config.region = 'us-west-2'
    # DeploymentSpec.DeploymentOperator.AWS_SAGEMAKER
    test_deployment_pb.spec.operator = 2

    deployment_operator = SageMakerDeploymentOperator()
    fake_yatai_service = MagicMock()
    fake_yatai_service.GetBento = lambda uri: mock_get_bento()
    result_pb = deployment_operator.apply(test_deployment_pb, fake_yatai_service)
    assert result_pb.status.status_code == Status.OK
    assert result_pb.deployment.name == 'test'
