import logging
import re
import subprocess
import os
import shutil

import boto3
from botocore.exceptions import ClientError

from bentoml.exceptions import (
    BentoMLException,
    MissingDependencyException,
    InvalidArgument,
)

logger = logging.getLogger(__name__)


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
