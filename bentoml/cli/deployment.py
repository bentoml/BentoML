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

import click
import logging

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    _echo,
    CLI_COLOR_SUCCESS,
    parse_yaml_file_callback,
)
from bentoml.yatai.deployment import ALL_NAMESPACE_TAG
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.exceptions import CLIException
from bentoml.cli.utils import Spinner, _print_deployment_info, _print_deployments_info
from bentoml.utils import get_default_yatai_client

yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')

# pylint: disable=unused-variable

logger = logging.getLogger(__name__)


DEFAULT_SAGEMAKER_INSTANCE_TYPE = 'ml.m4.xlarge'
DEFAULT_SAGEMAKER_INSTANCE_COUNT = 1


def get_deployment_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        help='Commands for managing and operating BentoService deployments',
        cls=BentoMLCommandGroup,
    )
    def deployment():
        pass

    @deployment.command(help='Create BentoService deployment from yaml file')
    @click.option(
        '-f',
        '--file',
        'deployment_yaml',
        type=click.File('r'),
        required=True,
        callback=parse_yaml_file_callback,
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def create(deployment_yaml, output, wait):
        yatai_client = get_default_yatai_client()
        deployment_name = deployment_yaml.get('name')
        with Spinner('Creating deployment '):
            result = yatai_client.deployment.create(deployment_yaml, wait)
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _echo(
            f'Successfully created deployment {deployment_name}', CLI_COLOR_SUCCESS,
        )
        _print_deployment_info(result.deployment, output)

    @deployment.command(help='Apply BentoService deployment from yaml file')
    @click.option(
        '-f',
        '--file',
        'deployment_yaml',
        type=click.File('r'),
        required=True,
        callback=parse_yaml_file_callback,
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def apply(deployment_yaml, output, wait):
        deployment_name = deployment_yaml.get('name')
        yatai_client = get_default_yatai_client()
        with Spinner('Applying deployment'):
            result = yatai_client.deployment.apply(deployment_yaml, wait)
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _echo(
            f'Successfully applied deployment {deployment_name}', CLI_COLOR_SUCCESS,
        )
        _print_deployment_info(result.deployment, output)

    @deployment.command(help='Delete deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "default" '
        'which can be changed in BentoML configuration file',
    )
    @click.option(
        '--force',
        is_flag=True,
        help='force delete the deployment record in database and '
        'ignore errors when deleting cloud resources',
    )
    def delete(name, namespace, force):
        yatai_client = get_default_yatai_client()
        get_deployment_result = yatai_client.deployment.get(namespace, name)
        if get_deployment_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        result = yatai_client.deployment.delete(name, namespace, force)
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _echo('Successfully deleted deployment "{}"'.format(name), CLI_COLOR_SUCCESS)

    @deployment.command(help='Get deployment information')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which '
        'can be changed in BentoML configuration file',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    def get(name, namespace, output):
        yatai_client = get_default_yatai_client()
        get_result = yatai_client.deployment.get(namespace, name)
        if get_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        describe_result = yatai_client.deployment.describe(
            namespace=namespace, name=name
        )
        if describe_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                describe_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        get_result.deployment.state.CopyFrom(describe_result.state)
        _print_deployment_info(get_result.deployment, output)

    @deployment.command(name='list', help='List deployments')
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which '
        'can be changed in BentoML configuration file',
        default=ALL_NAMESPACE_TAG,
    )
    @click.option(
        '-p', '--platform', type=click.Choice(['sagemaker', 'lambda']), help='platform',
    )
    @click.option(
        '--limit',
        type=click.INT,
        help='The maximum amount of deployments to be listed at once',
    )
    @click.option(
        '--labels',
        type=click.STRING,
        help="Label query to filter deployments, supports '=', '!=', 'IN', 'NotIn', "
        "'Exists', and 'DoesNotExist'. (e.g. key1=value1, key2!=value2, key3 "
        "In (value3, value3a), key4 DoesNotExist)",
    )
    @click.option(
        '--order-by', type=click.Choice(['created_at', 'name']), default='created_at',
    )
    @click.option(
        '--asc/--desc',
        default=False,
        help='Ascending or descending order for list deployments',
    )
    @click.option(
        '-o',
        '--output',
        type=click.Choice(['json', 'yaml', 'table', 'wide']),
        default='table',
    )
    def list_deployments(namespace, platform, limit, labels, order_by, asc, output):
        yatai_client = get_default_yatai_client()
        list_result = yatai_client.deployment.list(
            limit=limit,
            labels=labels,
            namespace=namespace,
            operator=platform,
            order_by=order_by,
            ascending_order=asc,
        )
        if list_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                list_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _print_deployments_info(list_result.deployments, output)

    return deployment
