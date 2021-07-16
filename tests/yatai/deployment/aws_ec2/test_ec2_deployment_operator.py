import os

from mock import MagicMock, patch

from bentoml.yatai.proto import status_pb2
from bentoml.yatai.proto.deployment_pb2 import (
    Deployment,
    DeploymentState,
    DescribeDeploymentResponse,
)
from bentoml.yatai.proto.repository_pb2 import (
    Bento,
    BentoServiceMetadata,
    BentoUri,
    GetBentoResponse,
)
from bentoml.yatai.deployment.aws_ec2.operator import (
    _make_cloudformation_template,
    AwsEc2DeploymentOperator,
)
from bentoml.yatai.deployment.aws_utils import FAILED_CLOUDFORMATION_STACK_STATUS

mock_s3_bucket_name = "test_deployment_bucket"
mock_s3_prefix = "prefix"
mock_s3_path = "s3://{}/{}".format(mock_s3_bucket_name, mock_s3_prefix)
mock_registry_name = "test_registry"
mock_registry_id = "7520142243238"
mock_elb_name = "elb-test"
mock_region = "us-east-1"
mock_registry_auth_token = "riqpoweripqwreweropi"
mock_registry_endpoint = (
    "https://752014255238.dkr.ecr.ap-south-1.amazonaws.com/bento-iris"
)
mock_target_group_arn = "target-us-east-1-aws"
mock_url = "http://mock-url.com"
mock_port_number = 123
mock_user_id = 1234567891


def create_yatai_service_mock(repo_storage_type=BentoUri.LOCAL):
    bento_pb = Bento(name="bento_test_name", version="version1.1.1")
    if repo_storage_type == BentoUri.LOCAL:
        bento_pb.uri.uri = "/tmp/path/to/bundle"
    bento_pb.uri.type = repo_storage_type
    api = BentoServiceMetadata.BentoServiceApi(name="predict")
    bento_pb.bento_service_metadata.apis.extend([api])
    bento_pb.bento_service_metadata.env.python_version = "3.7.0"
    get_bento_response = GetBentoResponse(bento=bento_pb)

    yatai_service_mock = MagicMock()
    yatai_service_mock.GetBento.return_value = get_bento_response
    return yatai_service_mock


def generate_ec2_deployment_pb():
    test_deployment_pb = Deployment(name="test_aws_ec2", namespace="test-namespace")
    test_deployment_pb.spec.bento_name = "bento_name"
    test_deployment_pb.spec.bento_version = "v1.0.0"
    # DeploymentSpec.DeploymentOperator.AWS_LAMBDA
    test_deployment_pb.spec.operator = 3
    test_deployment_pb.spec.aws_ec2_operator_config.region = "us-west-2"
    test_deployment_pb.spec.aws_ec2_operator_config.instance_type = "t2.micro"
    test_deployment_pb.spec.aws_ec2_operator_config.ami_id = "test-ami-id"
    test_deployment_pb.spec.aws_ec2_operator_config.autoscale_min_size = 1
    test_deployment_pb.spec.aws_ec2_operator_config.autoscale_desired_capacity = 1
    test_deployment_pb.spec.aws_ec2_operator_config.autoscale_max_size = 1

    return test_deployment_pb


def test_make_cloudformation_template(tmpdir):
    mock_template_name = "template.yaml"
    _make_cloudformation_template(
        tmpdir,
        "test_user_data",
        mock_s3_bucket_name,
        mock_template_name,
        mock_elb_name,
        "test_ami",
        "t2.micro",
        1,
        2,
        3,
    )
    assert os.path.isfile(os.path.join(tmpdir, mock_template_name))


@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_sam_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_docker_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.create_s3_bucket_if_not_exists",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.deploy_ec2_service", MagicMock(),
)
def test_ec2_add_success():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "GetCallerIdentity":
            return {"Account": mock_user_id}

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.add(test_deployment_pb)

    assert result_pb.status.status_code == status_pb2.Status.OK
    assert result_pb.deployment.state.state == DeploymentState.PENDING


def test_ec2_delete_success():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DeleteStack":
            return {}
        elif op_name == "DeleteRepository":
            return {}

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.delete(test_deployment_pb)
        assert result_pb.status.status_code == status_pb2.Status.OK


def test_ec2_describe_no_scaling_success():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {
                "Stacks": [
                    {
                        "StackStatus": "CREATE_COMPLETE",
                        "Outputs": [
                            {
                                "OutputKey": "S3Bucket",
                                "OutputValue": mock_s3_bucket_name,
                            },
                            {
                                "OutputKey": "TargetGroup",
                                "OutputValue": mock_target_group_arn,
                            },
                            {"OutputKey": "Url", "OutputValue": mock_url},
                        ],
                    }
                ]
            }
        if op_name == "DescribeTargetHealth":
            return {
                "TargetHealthDescriptions": [
                    {
                        "Target": {
                            "Id": "id-instance-1",
                            "Port": mock_port_number,
                            "AvailabilityZone": "us-east-1a",
                        },
                        "HealthCheckPort": "string",
                        "TargetHealth": {
                            "State": "healthy",
                            "Description": "mock-string",
                        },
                    },
                ]
            }

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.describe(test_deployment_pb)
        assert result_pb.status.status_code == status_pb2.Status.OK
        assert result_pb.state.state == DeploymentState.RUNNING


def test_ec2_describe_pending():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {
                "Stacks": [
                    {
                        "StackStatus": "STACK_UPDATING",
                        "Outputs": [
                            {
                                "OutputKey": "S3Bucket",
                                "OutputValue": mock_s3_bucket_name,
                            },
                            {
                                "OutputKey": "TargetGroup",
                                "OutputValue": mock_target_group_arn,
                            },
                            {"OutputKey": "Url", "OutputValue": mock_url},
                        ],
                    }
                ]
            }
        if op_name == "DescribeTargetHealth":
            return {
                "TargetHealthDescriptions": [
                    {
                        "Target": {
                            "Id": "id-instance-1",
                            "Port": mock_port_number,
                            "AvailabilityZone": "us-east-1a",
                        },
                        "HealthCheckPort": "string",
                        "TargetHealth": {
                            "State": "unhealthy",
                            "Description": "mock-string",
                        },
                    },
                ]
            }

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.describe(test_deployment_pb)
        assert result_pb.status.status_code == status_pb2.Status.OK
        assert result_pb.state.state == DeploymentState.PENDING


def test_ec2_describe_stack_failure():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {
                "Stacks": [
                    {
                        "StackStatus": FAILED_CLOUDFORMATION_STACK_STATUS[0],
                        "Outputs": [
                            {
                                "OutputKey": "S3Bucket",
                                "OutputValue": mock_s3_bucket_name,
                            },
                            {
                                "OutputKey": "TargetGroup",
                                "OutputValue": mock_target_group_arn,
                            },
                            {"OutputKey": "Url", "OutputValue": mock_url},
                        ],
                    }
                ]
            }
        if op_name == "DescribeTargetHealth":
            return {
                "TargetHealthDescriptions": [
                    {
                        "Target": {
                            "Id": "id-instance-1",
                            "Port": mock_port_number,
                            "AvailabilityZone": "us-east-1a",
                        },
                        "HealthCheckPort": "string",
                        "TargetHealth": {
                            "State": "healthy",
                            "Description": "mock-string",
                        },
                    },
                ]
            }

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.describe(test_deployment_pb)
        assert result_pb.status.status_code == status_pb2.Status.OK
        assert result_pb.state.state == DeploymentState.FAILED


def test_ec2_describe_no_bucket_failure():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {"Stacks": [{"StackStatus": "CREATE_COMPLETE"}]}

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.describe(test_deployment_pb)
        assert result_pb.status.status_code == status_pb2.Status.ABORTED
        assert result_pb.state.state == DeploymentState.ERROR


@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_sam_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_docker_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.deploy_ec2_service", MagicMock(),
)
def test_ec2_update_success():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {
                "Stacks": [
                    {
                        "StackStatus": "CREATE_COMPLETE",
                        "Outputs": [
                            {
                                "OutputKey": "S3Bucket",
                                "OutputValue": mock_s3_bucket_name,
                            },
                            {
                                "OutputKey": "TargetGroup",
                                "OutputValue": mock_target_group_arn,
                            },
                            {"OutputKey": "Url", "OutputValue": mock_url},
                        ],
                    }
                ]
            }
        if op_name == "DescribeTargetHealth":
            return {
                "TargetHealthDescriptions": [
                    {
                        "Target": {
                            "Id": "id-instance-1",
                            "Port": mock_port_number,
                            "AvailabilityZone": "us-east-1a",
                        },
                        "HealthCheckPort": "string",
                        "TargetHealth": {
                            "State": "healthy",
                            "Description": "mock-string",
                        },
                    },
                ]
            }

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.update(test_deployment_pb, test_deployment_pb)

    assert result_pb.status.status_code == status_pb2.Status.OK
    assert result_pb.deployment.state.state == DeploymentState.PENDING


@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_sam_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_docker_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.deploy_ec2_service", MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.AwsEc2DeploymentOperator.describe",
    MagicMock(
        return_value=DescribeDeploymentResponse(
            status=status_pb2.Status(
                status_code=status_pb2.Status.INTERNAL, error_message="failed"
            )
        )
    ),
)
def test_ec2_update_describe_failure():
    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    result_pb = operator.update(test_deployment_pb, test_deployment_pb)
    assert result_pb.status.status_code == status_pb2.Status.INTERNAL
    assert result_pb.deployment.state.state == DeploymentState.ERROR


@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_sam_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.ensure_docker_available_or_raise",
    MagicMock(),
)
@patch(
    "bentoml.yatai.deployment.aws_ec2.operator.deploy_ec2_service", MagicMock(),
)
def test_ec2_update_no_bucket_failure():
    def mock_boto_client(self, op_name, args):  # pylint: disable=unused-argument
        if op_name == "DescribeStacks":
            return {"Stacks": [{"StackStatus": "CREATE_COMPLETE"}]}

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_ec2_deployment_pb()
    operator = AwsEc2DeploymentOperator(yatai_service_mock)

    with patch("botocore.client.BaseClient._make_api_call", new=mock_boto_client):
        result_pb = operator.update(test_deployment_pb, test_deployment_pb)

    assert result_pb.status.status_code == status_pb2.Status.INTERNAL
    assert result_pb.deployment.state.state == DeploymentState.ERROR
