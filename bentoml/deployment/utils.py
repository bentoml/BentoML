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

from bentoml.exceptions import (
    BentoMLException,
    BentoMLMissingDependencyException,
    BentoMLInvalidArgumentException,
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
    # for FileNotFoundError doesn't exist in py2.7. check_output raise OSError instead
    import six

    if six.PY3:
        not_found_error = FileNotFoundError
    else:
        not_found_error = OSError

    try:
        subprocess.check_output(['docker', 'info'])
    except subprocess.CalledProcessError as error:
        raise BentoMLException(
            'Error executing docker command: {}'.format(error.output)
        )
    except not_found_error:
        raise BentoMLMissingDependencyException(
            'Docker is required for this deployment. Please visit '
            'www.docker.com for instructions'
        )


def raise_if_api_names_not_found_in_bento_service_metadata(metadata, api_names):
    all_api_names = [api.name for api in metadata.apis]

    if not set(api_names).issubset(all_api_names):
        raise BentoMLInvalidArgumentException(
            "Expect api names {api_names} to be "
            "subset of {all_api_names}".format(
                api_names=api_names, all_api_names=all_api_names
            )
        )


def exception_to_return_status(error):
    from bentoml.yatai.status import Status

    if type(error) is BentoMLInvalidArgumentException:
        return Status.INVALID_ARGUMENT(str(error))
    else:
        return Status.INTERNAL(str(error))


def generate_aws_compatible_string(item):
    pattern = re.compile("[^a-zA-Z0-9-]|_")
    name = re.sub(pattern, "-", item)
    if len(name) > 63:
        raise BentoMLException(
            'AWS string {} exceeds maximum length of 63'.format(name)
        )
    return name
