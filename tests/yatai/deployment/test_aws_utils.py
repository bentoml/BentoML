import boto3
from mock import MagicMock, patch

from bentoml.yatai.deployment.aws_utils import create_ecr_repository_if_not_exists

mock_s3_bucket_name = 'test_deployment_bucket'
mock_s3_prefix = 'prefix'
mock_s3_path = 's3://{}/{}'.format(mock_s3_bucket_name, mock_s3_prefix)
mock_repository_name = "test_registry"
mock_repository_id = "7520142243238"
mock_elb_name = "elb-test"
mock_region = "us-east-1"
mock_repository_auth_token = "riqpoweripqwreweropi"
mock_repository_endpoint = (
    "https://752014255238.dkr.ecr.ap-south-1.amazonaws.com/bento-iris"
)


def test_create_ecr_repo():
    ecr_client = boto3.client("ecr", mock_region)

    def mock_ecr_create_success(
        self, op_name, args
    ):  # pylint: disable=unused-argument
        if op_name == 'CreateRepository':
            return {"repository": {"registryId": mock_repository_id}}
        elif op_name == 'DescribeRepositories':
            raise ecr_client.exceptions.RepositoryNotFoundException(
                operation_name='', error_response={}
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
        elif op_name == "DescribeRepositories":
            raise ecr_client.exceptions.RepositoryAlreadyExistsException(
                operation_name='', error_response={}
            )

    with patch(
        "botocore.client.BaseClient._make_api_call", new=mock_ecr_describe_success
    ):
        r = create_ecr_repository_if_not_exists(mock_region, mock_repository_name)
        assert r == mock_repository_id
