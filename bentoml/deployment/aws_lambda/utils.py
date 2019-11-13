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

import subprocess
import logging

from bentoml.exceptions import BentoMLException, BentoMLMissingDependencyException


logger = logging.getLogger(__name__)


def ensure_sam_available_or_raise():
    # for FileNotFoundError doesn't exist in py2.7. check_output raise OSError instead
    import six

    if six.PY3:
        not_found_error = FileNotFoundError
    else:
        not_found_error = OSError

    try:
        subprocess.check_output(['sam', '--version'])
    except subprocess.CalledProcessError as error:
        raise BentoMLException('Error executing sam command: {}'.format(error.output))
    except not_found_error:
        raise BentoMLMissingDependencyException(
            'SAM is required for AWS Lambda deployment. Please visit '
            'https://aws.amazon.com/serverless/sam for instructions'
        )


def call_sam_command(command):
    command = ['sam'] + command
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        stdout = proc.stdout.read().decode('utf-8')
        logger.debug('SAM cmd output: %s', stdout)
    return stdout

