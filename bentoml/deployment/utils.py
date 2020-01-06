# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import re
import subprocess

import boto3
from botocore.exceptions import ClientError

from bentoml.exceptions import (
    BentoMLException,
    MissingDependencyException,
    InvalidArgument,
)

logger = logging.getLogger(__name__)


def process_docker_api_line(payload):
    """ Process the output from API stream, throw an Exception if there is an error """
    # Sometimes Docker sends to "{}\n" blocks together...
    errors = []
    for segment in payload.decode("utf-8").strip().split("\n"):
        line = segment.strip()
        if line:
            try:
                line_payload = json.loads(line)
            except ValueError as e:
                logger.warning("Could not decipher payload from Docker API: %s", str(e))
            if line_payload:
                if "errorDetail" in line_payload:
                    error = line_payload["errorDetail"]
                    error_msg = 'Error running docker command: {}: {}'.format(
                        error["code"], error['message']
                    )
                    logger.error(error_msg)
                    errors.append(error_msg)
                elif "stream" in line_payload:
                    logger.info(line_payload['stream'])

    if errors:
        error_msg = ";".join(errors)
        raise BentoMLException("Error running docker command: {}".format(error_msg))


def ensure_docker_available_or_raise():
    try:
        subprocess.check_output(['docker', 'info'])
    except subprocess.CalledProcessError as error:
        raise BentoMLException(
            'Error executing docker command: {}'.format(error.output.decode())
        )
    except FileNotFoundError:
        raise MissingDependencyException(
            'Docker is required for this deployment. Please visit '
            'www.docker.com for instructions'
        )


def raise_if_api_names_not_found_in_bento_service_metadata(metadata, api_names):
    all_api_names = [api.name for api in metadata.apis]

    if not set(api_names).issubset(all_api_names):
        raise InvalidArgument(
            "Expect api names {api_names} to be "
            "subset of {all_api_names}".format(
                api_names=api_names, all_api_names=all_api_names
            )
        )


def generate_aws_compatible_string(*items, max_length=63):
    """
    Generate a AWS resource name that is composed from list of string items. This
    function replaces all invalid characters in the given items into '-', and allow user
    to specify the max_length for each part separately by passing the item and its max
    length in a tuple, e.g.:

    >> generate_aws_compatible_string("abc", "def")
    >> 'abc-def'  # concatenate mupltiple parts

    >> generate_aws_compatible_string("abc_def")
    >> 'abc-def'  # replace invalid chars to '-'

    >> generate_aws_compatible_string(("ab", 1), ("bcd", 2), max_length=4)
    >> 'a-bc'  # trim based on max_length of each part
    """
    trimed_items = [
        item[0][: item[1]] if type(item) == tuple else item for item in items
    ]
    items = [item[0] if type(item) == tuple else item for item in items]

    for i in range(len(trimed_items)):
        if len('-'.join(items)) <= max_length:
            break
        else:
            items[i] = trimed_items[i]

    name = '-'.join(items)
    if len(name) > max_length:
        raise BentoMLException(
            'AWS resource name {} exceeds maximum length of {}'.format(name, max_length)
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
        logger.error('Encounter error when getting default region for AWS: %s', str(e))
        return ''
