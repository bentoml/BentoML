from mock import patch, MagicMock

from botocore.exceptions import ClientError
from botocore.stub import Stubber

import botocore
import boto3

from bentoml.deployment.sagemaker import (
    _parse_aws_client_exception_or_raise,
    _cleanup_sagemaker_model,
    _cleanup_sagemaker_endpoint_config,
    init_sagemaker_project,
    get_arn_role_from_current_aws_user,
    SageMakerDeploymentOperator)
from bentoml.proto.deployment_pb2 import Deployment, DeploymentSpec
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
        return {
            'Arn': 'something:something:role/random'
        }
    elif operation_name == 'GetRole':
        return {
            'Role': {
                'Arn': 'arn:aws:us-west-2:999888'
            }
        }
    elif operation_name == 'GetAuthorizationToken':
        return {
            'authorizationData': [
                {
                    #b64encoded string of 'user:password'
                    'authorizationToken': 'dXNlcjpwYXNzd29yZA=='
                }
            ]
        }
    elif operation_name == 'DescribeRepositories':
        ecr_client = boto3.client('ecr')
        raise ecr_client.exceptions.RepositoryNotFoundException('message')
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
