import boto3
import pytest
from botocore.exceptions import ClientError
from mock import patch

from bentoml.exceptions import AWSServiceError, BentoMLException
from bentoml.yatai.deployment.aws_utils import (
    create_ecr_repository_if_not_exists,
    get_ecr_login_info,
    generate_bentoml_exception_from_aws_client_error,
    describe_cloudformation_stack,
)

mock_s3_bucket_name = "test_deployment_bucket"
mock_s3_prefix = "prefix"
mock_s3_path = "s3://{}/{}".format(mock_s3_bucket_name, mock_s3_prefix)
mock_repository_name = "test_registry"
mock_repository_id = "7520142243238"
mock_repository_username = "abc"
mock_repository_password = "123"
mock_elb_name = "elb-test"
mock_region = "us-east-1"
mock_repository_auth_token = "YWJjOjEyMw=="
mock_repository_endpoint = (
    "https://752014255238.dkr.ecr.ap-south-1.amazonaws.com/bento-iris"
)
mock_stack_name = "mock_stack"
mock_stack_info = "abc"


def test_create_ecr_repository_if_not_exists():
    ecr_client = boto3.client("ecr", mock_region)

    def mock_ecr_create_success(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "CreateRepository":
            return {"repository": {"registryId": mock_repository_id}}
        elif op_name == "DescribeRepositories":
            raise ecr_client.exceptions.RepositoryNotFoundException(
                operation_name="", error_response={}
            )

    with patch(
        "botocore.client.BaseClient._make_api_call", new=mock_ecr_create_success
    ):
        r = create_ecr_repository_if_not_exists(mock_region, mock_repository_name)
        assert r == mock_repository_id

    def mock_ecr_describe_success(
        self, op_name, args
    ):  # pylint: disable=unused-argument
        if op_name == "DescribeRepositories":
            return {"repositories": [{"registryId": mock_repository_id}]}
        elif op_name == "CreateRepository":
            raise ecr_client.exceptions.RepositoryAlreadyExistsException(
                operation_name="", error_response={}
            )

    with patch(
        "botocore.client.BaseClient._make_api_call", new=mock_ecr_describe_success
    ):
        r = create_ecr_repository_if_not_exists(mock_region, mock_repository_name)
        assert r == mock_repository_id


def test_get_ecr_login_info():
    def mock_ecr_client_auth_token(
        self, op_name, args
    ):  # pylint: disable=unused-argument
        if op_name == "GetAuthorizationToken":
            return {
                "authorizationData": [
                    {
                        "authorizationToken": mock_repository_auth_token,
                        "proxyEndpoint": mock_repository_endpoint,
                    }
                ]
            }

    with patch(
        "botocore.client.BaseClient._make_api_call", new=mock_ecr_client_auth_token
    ):
        registry_url, username, password = get_ecr_login_info(
            mock_region, mock_repository_id
        )
        assert registry_url == mock_repository_endpoint
        assert username == mock_repository_username
        assert password == mock_repository_password


def test_generate_bentoml_exception_from_aws_client_error():
    error = ClientError(
        error_response={
            "Error": {"Code": "ValidationException", "Message": "error message"}
        },
        operation_name="failed_operation",
    )
    exception = generate_bentoml_exception_from_aws_client_error(error)
    assert isinstance(exception, AWSServiceError)


def test_describe_cloudformation_stack():
    def mock_cf_client_success(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {"Stacks": [mock_stack_info]}

    with patch("botocore.client.BaseClient._make_api_call", new=mock_cf_client_success):
        stack_result = describe_cloudformation_stack(mock_region, mock_stack_name)
        assert stack_result == mock_stack_info

    def mock_cf_client_too_many(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {"Stacks": [mock_stack_info, "another stack"]}

    with patch(
        "botocore.client.BaseClient._make_api_call", new=mock_cf_client_too_many
    ):
        with pytest.raises(BentoMLException) as error:
            describe_cloudformation_stack(mock_region, mock_stack_name)
        assert f"Found more than one cloudformation stack for {mock_stack_name}" in str(
            error.value
        )

    with patch("botocore.client.BaseClient._make_api_call") as mock_client_call:
        mock_client_call.side_effect = ClientError(
            operation_name="DescribeStacks", error_response={}
        )
        with pytest.raises(BentoMLException) as error:
            describe_cloudformation_stack(mock_region, mock_stack_name)
        assert f"Failed to describe CloudFormation {mock_stack_name}" in str(
            error.value
        )
