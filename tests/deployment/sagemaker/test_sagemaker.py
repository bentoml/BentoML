from botocore.exceptions import ClientError
from botocore.stub import Stubber

import boto3

from bentoml.deployment.sagemaker import _parse_aws_client_exception_or_raise
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
