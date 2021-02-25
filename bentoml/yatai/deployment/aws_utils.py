import base64
import logging
import re
import subprocess
import os

import boto3
from botocore.exceptions import ClientError

from bentoml.exceptions import (
    BentoMLException,
    MissingDependencyException,
    AWSServiceError,
)

logger = logging.getLogger(__name__)

# https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/\
# using-cfn-describing-stacks.html
FAILED_CLOUDFORMATION_STACK_STATUS = [
    "CREATE_FAILED",
    # Ongoing creation of one or more stacks with an expected StackId
    # but without any templates or resources.
    "REVIEW_IN_PROGRESS",
    "ROLLBACK_FAILED",
    # This status exists only after a failed stack creation.
    "ROLLBACK_COMPLETE",
    # Ongoing removal of one or more stacks after a failed stack
    # creation or after an explicitly cancelled stack creation.
    "ROLLBACK_IN_PROGRESS",
]

SUCCESS_CLOUDFORMATION_STACK_STATUS = ["CREATE_COMPLETE", "UPDATE_COMPLETE"]


def generate_aws_compatible_string(*items, max_length=63):
    """
    Generate a AWS resource name that is composed from list of string items. This
    function replaces all invalid characters in the given items into '-', and allow user
    to specify the max_length for each part separately by passing the item and its max
    length in a tuple, e.g.:

    >> generate_aws_compatible_string("abc", "def")
    >> 'abc-def'  # concatenate multiple parts

    >> generate_aws_compatible_string("abc_def")
    >> 'abc-def'  # replace invalid chars to '-'

    >> generate_aws_compatible_string(("ab", 1), ("bcd", 2), max_length=4)
    >> 'a-bc'  # trim based on max_length of each part
    """
    trimmed_items = [
        item[0][: item[1]] if type(item) == tuple else item for item in items
    ]
    items = [item[0] if type(item) == tuple else item for item in items]

    for i in range(len(trimmed_items)):
        if len("-".join(items)) <= max_length:
            break
        else:
            items[i] = trimmed_items[i]

    name = "-".join(items)
    if len(name) > max_length:
        raise BentoMLException(
            "AWS resource name {} exceeds maximum length of {}".format(name, max_length)
        )
    invalid_chars = re.compile("[^a-zA-Z0-9-]|_")
    name = re.sub(invalid_chars, "-", name)
    return name


def get_default_aws_region():
    try:
        aws_session = boto3.session.Session()
        region = aws_session.region_name
        if not region:
            return ""
        return aws_session.region_name
    except ClientError as e:
        # We will do nothing, if there isn't a default region
        logger.error("Encounter error when getting default region for AWS: %s", str(e))
        return ""


def ensure_sam_available_or_raise():
    try:
        import samcli

        if samcli.__version__ != "0.33.1":
            raise BentoMLException(
                "aws-sam-cli package requires version 0.33.1 "
                "Install the package with `pip install -U aws-sam-cli==0.33.1`"
            )
    except ImportError:
        raise MissingDependencyException(
            "aws-sam-cli package is required. Install "
            "with `pip install --user aws-sam-cli`"
        )


def call_sam_command(command, project_dir, region):
    command = ["sam"] + command

    # We are passing region as part of the param, due to sam cli is not currently
    # using the region that passed in each command.  Set the region param as
    # AWS_DEFAULT_REGION for the subprocess call
    logger.debug('Setting envar "AWS_DEFAULT_REGION" to %s for subprocess call', region)
    copied_env = os.environ.copy()
    copied_env["AWS_DEFAULT_REGION"] = region

    proc = subprocess.Popen(
        command,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=copied_env,
    )
    stdout, stderr = proc.communicate()
    logger.debug("SAM cmd %s output: %s", command, stdout.decode("utf-8"))
    return proc.returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def validate_sam_template(template_file, aws_region, sam_project_path):
    status_code, stdout, stderr = call_sam_command(
        ["validate", "--template-file", template_file, "--region", aws_region],
        project_dir=sam_project_path,
        region=aws_region,
    )
    if status_code != 0:
        error_message = stderr
        if not error_message:
            error_message = stdout
        raise BentoMLException(
            "Failed to validate lambda template. {}".format(error_message)
        )


def cleanup_s3_bucket_if_exist(bucket_name, region):
    s3_client = boto3.client('s3', region)
    s3 = boto3.resource('s3')
    try:
        logger.debug('Removing all objects inside bucket %s', bucket_name)
        s3.Bucket(bucket_name).objects.all().delete()
        logger.debug('Deleting bucket %s', bucket_name)
        s3_client.delete_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response and e.response['Error']['Code'] == 'NoSuchBucket':
            # If there is no bucket, we just let it silently fail, dont have to do
            # any thing
            return
        else:
            raise e


def delete_cloudformation_stack(stack_name, region):
    cf_client = boto3.client("cloudformation", region)
    cf_client.delete_stack(StackName=stack_name)


def delete_ecr_repository(repository_name, region):
    try:
        ecr_client = boto3.client("ecr", region)
        ecr_client.delete_repository(repositoryName=repository_name, force=True)
    except ClientError as e:
        if e.response and e.response['Error']['Code'] == 'RepositoryNotFoundException':
            # Don't raise error, if the repo can't be found
            return
        else:
            raise e


def get_instance_public_ip(instance_id, region):
    ec2_client = boto3.client("ec2", region)
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    all_instances = response["Reservations"][0]["Instances"]
    if all_instances:
        if "PublicIpAddress" in all_instances[0]:
            return all_instances[0]["PublicIpAddress"]
    return ""


def get_instance_ip_from_scaling_group(autoscaling_group_names, region):
    asg_client = boto3.client("autoscaling", region)
    response = asg_client.describe_auto_scaling_groups(
        AutoScalingGroupNames=autoscaling_group_names
    )
    all_autoscaling_group_info = response["AutoScalingGroups"]

    all_instances = []
    if all_autoscaling_group_info:
        for group in all_autoscaling_group_info:
            for instance in group["Instances"]:
                endpoint = get_instance_public_ip(instance["InstanceId"], region)
                all_instances.append(
                    {
                        "instance_id": instance["InstanceId"],
                        "endpoint": endpoint,
                        "state": instance["LifecycleState"],
                        "health_status": instance["HealthStatus"],
                    }
                )

    return all_instances


def get_aws_user_id():
    return boto3.client("sts").get_caller_identity().get("Account")


def create_ecr_repository_if_not_exists(region, repository_name):
    ecr_client = boto3.client("ecr", region)
    try:
        result = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repository_id = result['repositories'][0]['registryId']
    except ecr_client.exceptions.RepositoryNotFoundException:
        result = ecr_client.create_repository(repositoryName=repository_name)
        repository_id = result['repository']['registryId']
    return repository_id


def get_ecr_login_info(region, repository_id):
    ecr_client = boto3.client('ecr', region)
    token = ecr_client.get_authorization_token(registryIds=[repository_id])
    logger.debug("Getting docker login info from AWS")
    username, password = (
        base64.b64decode(token["authorizationData"][0]["authorizationToken"])
        .decode("utf-8")
        .split(":")
    )
    registry_url = token["authorizationData"][0]["proxyEndpoint"]

    return registry_url, username, password


def generate_bentoml_exception_from_aws_client_error(e, message_prefix=None):
    """parse botocore.exceptions.ClientError into Bento StatusProto

     We handle two most common errors when deploying to Sagemaker.
         1. Authentication issue/invalid access(InvalidSignatureException)
         2. resources not found (ValidationException)
     It will return correlated StatusProto(NOT_FOUND, UNAUTHENTICATED)

     Args:
         e: ClientError from botocore.exceptions
         message_prefix: string
     Returns:
         StatusProto
     """
    error_response = e.response.get("Error", {})
    error_code = error_response.get("Code", "Unknown")
    error_message = error_response.get("Message", "Unknown")
    error_log_message = (
        f"AWS ClientError - operation: {e.operation_name}, "
        f"code: {error_code}, message: {error_message}"
    )
    if message_prefix:
        error_log_message = f"{message_prefix}; {error_log_message}"
    logger.error(error_log_message)
    return AWSServiceError(error_log_message)


def describe_cloudformation_stack(region, stack_name):
    cf_client = boto3.client("cloudformation", region)
    try:
        cloudformation_stack_result = cf_client.describe_stacks(StackName=stack_name)
        stack_info = cloudformation_stack_result.get('Stacks')
        if len(stack_info) < 1:
            raise BentoMLException(f'Cloudformation {stack_name} not found')
        if len(stack_info) > 1:
            raise BentoMLException(
                f'Found more than one cloudformation stack for {stack_name}'
            )
        return stack_info[0]
    except ClientError as error:
        raise BentoMLException(
            f'Failed to describe CloudFormation {stack_name} {error}'
        )
