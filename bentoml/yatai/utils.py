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

import subprocess
import urllib3

from bentoml.exceptions import BentoMLException, MissingDependencyException


def ensure_node_available_or_raise():
    try:
        subprocess.check_output(['node', '--version'])
    except subprocess.CalledProcessError as error:
        raise BentoMLException(
            'Error executing node command: {}'.format(error.output.decode())
        )
    except FileNotFoundError:
        raise MissingDependencyException(
            'Node is required for Yatai web UI. Please visit '
            'www.nodejs.org for instructions'
        )


def parse_grpc_url(url):
    '''
    >>> parse_grpc_url("grpcs://yatai.com:43/query")
    ('grpcs', 'yatai.com:43/query')
    >>> parse_grpc_url("yatai.com:43/query")
    (None, 'yatai.com:43/query')
    '''
    parts = urllib3.util.parse_url(url)
    return parts.scheme, url.replace(f"{parts.scheme}://", "", 1)
