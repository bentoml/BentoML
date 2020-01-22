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

import click
import logging
from datetime import datetime

from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    _echo,
    CLI_COLOR_ERROR,
    CLI_COLOR_SUCCESS,
    parse_yaml_file_callback,
)
from bentoml.yatai.client import YataiClient
from bentoml.deployment.store import ALL_NAMESPACE_TAG
from bentoml.proto.deployment_pb2 import DeploymentSpec
from bentoml.proto import status_pb2
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.usage_stats import track_cli
from bentoml.exceptions import BentoMLException
from bentoml.cli.utils import Spinner, _print_deployment_info, _print_deployments_info

# pylint: disable=unused-variable

logger = logging.getLogger(__name__)


DEFAULT_SAGEMAKER_INSTANCE_TYPE = 'ml.m4.xlarge'
DEFAULT_SAGEMAKER_INSTANCE_COUNT = 1


def get_deployment_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        help='Commands for manageing and operating BentoService deployments',
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
        yatai_client = YataiClient()
        platform_name = deployment_yaml.get('spec', {}).get('operator')
        deployment_name = deployment_yaml.get('name')
        track_cli('deploy-create', platform_name)
        try:
            with Spinner('Creating deployment '):
                result = yatai_client.deployment.create(deployment_yaml, wait)
            if result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                track_cli(
                    'deploy-create-failure',
                    platform_name,
                    {'error_code': error_code, 'error_message': error_message},
                )
                _echo(
                    f'Failed to create deployment {deployment_name} '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            track_cli('deploy-create-success', platform_name)
            _echo(
                f'Successfully created deployment {deployment_name}', CLI_COLOR_SUCCESS,
            )
            _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                f'Failed to create deployment {deployment_name} {str(e)}',
                CLI_COLOR_ERROR,
            )

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
        track_cli('deploy-apply', deployment_yaml.get('spec', {}).get('operator'))
        platform_name = deployment_yaml.get('spec', {}).get('operator')
        deployment_name = deployment_yaml.get('name')
        try:
            yatai_client = YataiClient()
            with Spinner('Applying deployment'):
                result = yatai_client.deployment.apply(deployment_yaml, wait)
            if result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                track_cli(
                    'deploy-apply-failure',
                    platform_name,
                    {'error_code': error_code, 'error_message': error_message},
                )
                _echo(
                    f'Failed to apply deployment {deployment_name} '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            track_cli('deploy-create-success', platform_name)
            _echo(
                f'Successfully applied deployment {deployment_name}', CLI_COLOR_SUCCESS,
            )
            _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            track_cli(
                'deploy-apply-failure', platform_name, {'error_message': str(e)},
            )
            _echo(
                'Failed to apply deployment {name}. Error message: {message}'.format(
                    name=deployment_yaml.get('name'), message=e
                )
            )

    @deployment.command(help='Delete deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "default" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '--force',
        is_flag=True,
        help='force delete the deployment record in database and '
        'ignore errors when deleting cloud resources',
    )
    def delete(name, namespace, force):
        yatai_client = YataiClient()
        get_deployment_result = yatai_client.deployment.get(namespace, name)
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            _echo(
                f'Failed to get deployment {name} for deletion. '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        platform = DeploymentSpec.DeploymentOperator.Name(
            get_deployment_result.deployment.spec.operator
        )
        track_cli('deploy-delete', platform)
        result = yatai_client.deployment.delete(name, namespace, force)
        if result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            _echo(
                f'Failed to delete deployment {name}. {error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        extra_properties = {}
        if get_deployment_result.deployment.created_at:
            stopped_time = datetime.utcnow()
            extra_properties['uptime'] = int(
                (
                    stopped_time
                    - get_deployment_result.deployment.created_at.ToDatetime()
                ).total_seconds()
            )
        track_cli('deploy-delete-success', platform, extra_properties)
        _echo('Successfully deleted deployment "{}"'.format(name), CLI_COLOR_SUCCESS)

    @deployment.command(help='Get deployment information')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    def get(name, namespace, output):
        yatai_client = YataiClient()
        track_cli('deploy-get')
        get_result = yatai_client.deployment.get(namespace, name)
        if get_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_result.status
            )
            _echo(
                f'Failed to get deployment {name}. ' f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        describe_result = yatai_client.deployment.describe(
            namespace=namespace, name=name
        )
        if describe_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                describe_result.status
            )
            _echo(
                f'Failed to retrieve the latest status for deployment'
                f' {name}. {error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        get_result.deployment.state.CopyFrom(describe_result.state)
        _print_deployment_info(get_result.deployment, output)

    @deployment.command(name='list', help='List deployments')
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
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
        '-l',
        '--labels',
        type=click.STRING,
        help='List deployments matching the giving labels',
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
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table']), default='table'
    )
    def list_deployments(namespace, platform, limit, labels, order_by, asc, output):
        yatai_client = YataiClient()
        track_cli('deploy-list')
        try:
            list_result = yatai_client.deployment.list(
                limit=limit,
                labels=labels,
                namespace=namespace,
                operator=platform,
                order_by=order_by,
                ascending_order=asc,
            )
            if list_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    list_result.status
                )
                _echo(
                    f'Failed to list deployments {error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            _print_deployments_info(list_result.deployments, output)
        except BentoMLException as e:
            _echo(f'Failed to list deployments {str(e)}')

    return deployment
