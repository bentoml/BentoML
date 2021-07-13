import boto3

from bentoml.yatai.deployment.aws_ec2.constants import (
    BENTOSERVICE_PORT,
    AWS_EC2_IN_SERVICE_STATE,
    TARGET_HEALTHY_STATUS,
)
from bentoml.yatai.deployment.aws_utils import call_sam_command
from bentoml.exceptions import BentoMLException


def build_template(template_file_path, project_directory, region):
    status_code, stdout, stderr = call_sam_command(
        ["build", "-t", template_file_path], project_directory, region
    )

    if status_code != 0:
        error_message = stderr if stderr else stdout
        raise BentoMLException("Failed to build ec2 service {}".format(error_message))

    return status_code, stdout, stderr


def package_template(s3_bucket_name, project_directory, region):
    status_code, stdout, stderr = call_sam_command(
        [
            "package",
            "--output-template-file",
            "packaged.yaml",
            "--s3-bucket",
            s3_bucket_name,
        ],
        project_directory,
        region,
    )
    if status_code != 0:
        error_message = stderr if stderr else stdout
        raise BentoMLException("Failed to package ec2 service {}".format(error_message))
    return status_code, stdout, stderr


def deploy_template(stack_name, s3_bucket_name, project_directory, region):
    status_code, stdout, stderr = call_sam_command(
        [
            "deploy",
            "--template-file",
            "packaged.yaml",
            "--stack-name",
            stack_name,
            "--capabilities",
            "CAPABILITY_IAM",
            "--s3-bucket",
            s3_bucket_name,
        ],
        project_directory,
        region,
    )
    if status_code != 0:
        error_message = stderr if stderr else stdout
        raise BentoMLException("Failed to deploy ec2 service {}".format(error_message))
    return status_code, stdout, stderr


def get_endpoints_from_instance_address(instances, api_names):
    all_endpoints = []
    for instance in instances:
        if instance["state"] == AWS_EC2_IN_SERVICE_STATE:
            for api in api_names:
                all_endpoints.append(
                    "{ep}:{port}/{api}".format(
                        ep=instance["endpoint"], port=BENTOSERVICE_PORT, api=api
                    )
                )

    return all_endpoints


def get_healthy_target(target_group_arn, region):
    eb_client = boto3.client("elbv2", region)

    all_targets = eb_client.describe_target_health(TargetGroupArn=target_group_arn)
    for instance in all_targets["TargetHealthDescriptions"]:
        if instance["TargetHealth"]["State"] == TARGET_HEALTHY_STATUS:
            return instance
    return None
