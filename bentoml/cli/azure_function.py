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

from datetime import datetime

import click

from bentoml.cli.utils import Spinner
from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    parse_bento_tag_callback,
    CLI_COLOR_SUCCESS,
    CLI_COLOR_ERROR,
    _echo,
    parse_labels_callback,
    validate_labels_query_callback,
)
from bentoml.cli.deployment import _print_deployment_info, _print_deployments_info
from bentoml.deployment.store import ALL_NAMESPACE_TAG
from bentoml.exceptions import BentoMLException
from bentoml.proto import status_pb2
from bentoml.proto.deployment_pb2 import DeploymentSpec
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient

PLATFORM_NAME = DeploymentSpec.DeploymentOperator.Name(DeploymentSpec.AZURE_FUNCTION)


def get_azure_function_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        name='azure-function',
        help='Commands for Azure function BentoService deployment',
        cls=BentoMLCommandGroup,
    )
    def azure_function():
        pass

    @azure_function.command(help='Deploy BentoService to Azure function')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-b',
        '--bento',
        '--bento-service-bundle',
        type=click.STRING,
        required=True,
        callback=parse_bento_tag_callback,
        help='',
    )
    def deploy(name, bento):
        pass

    @azure_function.command(help='Update existing Azure function deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-b',
        '--bento',
        '--bento-service-bundle',
        type=click.STRING,
        callback=parse_bento_tag_callback,
        help='Target BentoService to be deployed, referenced by its name and version '
        'in format of name:version. For example: "iris_classifier:v1.2.0"',
    )
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def update(name, namespace, bento, output, wait):
        pass

    @azure_function.command(help='Delete Azure functiond deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration yatai_service/default_namespace',
    )
    @click.option(
        '--force',
        is_flag=True,
        help='force delete the deployment record in database and '
        'ignore errors when deleting cloud resources',
    )
    def delete(name, namespace, force):
        pass

    @azure_function.command()
    def get():
        pass

    @azure_function.command(name='list')
    def list_deployment():
        pass

    return azure_function
