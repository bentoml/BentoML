from bentoml.yatai.deployment.aws_utils import call_sam_command
from bentoml.exceptions import BentoMLException


def _build_template(template_file_path, project_directory, region):
    status_code, stdout, stderr = call_sam_command(
        ["build", "-t", template_file_path], project_directory, region
    )

    if status_code != 0:
        error_message = stderr if stderr else stdout
        raise BentoMLException("Failed to deploy ec2 service {}".format(error_message))

    return status_code, stdout, stderr


def _package_template(s3_bucket_name, project_directory, region):
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
    return status_code, stdout, stderr


def _deploy_template(stack_name, s3_bucket_name, project_directory, region):
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
    return status_code, stdout, stderr
