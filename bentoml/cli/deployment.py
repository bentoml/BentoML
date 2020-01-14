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
import time
import json
from datetime import datetime

from google.protobuf.json_format import MessageToJson
from tabulate import tabulate
import humanfriendly

from bentoml.cli.click_utils import (
    _echo,
    CLI_COLOR_ERROR,
    CLI_COLOR_SUCCESS,
    parse_yaml_file_callback,
)
from bentoml.proto.deployment_pb2 import DeploymentSpec, DeploymentState
from bentoml.proto import status_pb2
from bentoml.utils import pb_to_yaml
from bentoml.utils.usage_stats import track_cli
from bentoml.exceptions import BentoMLException
from bentoml.cli.utils import Spinner, parse_pb_response_error_message
from bentoml.yatai.client import YataiClient

# pylint: disable=unused-variable

logger = logging.getLogger(__name__)


DEFAULT_SAGEMAKER_INSTANCE_TYPE = 'ml.m4.xlarge'
DEFAULT_SAGEMAKER_INSTANCE_COUNT = 1


def parse_key_value_pairs(key_value_pairs_str):
    result = {}
    if key_value_pairs_str:
        for key_value_pair in key_value_pairs_str.split(','):
            key, value = key_value_pair.split('=')
            key = key.strip()
            value = value.strip()
            if key in result:
                logger.warning("duplicated key '%s' found string map parameter", key)
            result[key] = value
    return result


def _print_deployment_info(deployment, output_type):
    if output_type == 'yaml':
        result = pb_to_yaml(deployment)
    else:
        result = MessageToJson(deployment)
        if deployment.state.info_json:
            result = json.loads(result)
            result['state']['infoJson'] = json.loads(deployment.state.info_json)
            _echo(json.dumps(result, indent=2, separators=(',', ': ')))
            return
    _echo(result)


def _format_labels_for_print(labels):
    if not labels:
        return None
    result = []
    for label_key in labels:
        result.append(
            '{label_key}:{label_value}'.format(
                label_key=label_key, label_value=labels[label_key]
            )
        )
    return '\n'.join(result)


def _format_deployment_age_for_print(deployment_pb):
    if not deployment_pb.created_at:
        # deployments created before version 0.4.5 don't have created_at field,
        # we will not show the age for those deployments
        return None
    else:
        deployment_duration = datetime.utcnow() - deployment_pb.created_at.ToDatetime()
        return humanfriendly.format_timespan(deployment_duration)


def _print_deployments_table(deployments):
    table = []
    headers = ['NAME', 'NAMESPACE', 'LABELS', 'PLATFORM', 'STATUS', 'AGE']
    for deployment in deployments:
        row = [
            deployment.name,
            deployment.namespace,
            _format_labels_for_print(deployment.labels),
            DeploymentSpec.DeploymentOperator.Name(deployment.spec.operator)
            .lower()
            .replace('_', '-'),
            DeploymentState.State.Name(deployment.state.state)
            .lower()
            .replace('_', ' '),
            _format_deployment_age_for_print(deployment),
        ]
        table.append(row)
    table_display = tabulate(table, headers, tablefmt='plain')
    _echo(table_display)


def _print_deployments_info(deployments, output_type):
    if output_type == 'table':
        _print_deployments_table(deployments)
    else:
        for deployment in deployments:
            _print_deployment_info(deployment, output_type)


def get_state_after_await_action_complete(
    yatai_client, name, namespace, message, timeout_limit=600, wait_time=5
):
    start_time = time.time()

    with Spinner(message):
        while (time.time() - start_time) < timeout_limit:
            result = yatai_client.deployment.describe(namespace, name)
            if (
                result.status.status_code == status_pb2.Status.OK
                and result.state.state is DeploymentState.PENDING
            ):
                time.sleep(wait_time)
                continue
            else:
                break
    return result


def add_additional_deployment_commands(cli):
    @cli.command()
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
    def deploy(deployment_yaml, output, wait):
        track_cli('deploy-deploy', deployment_yaml.get('spec', {}).get('operator'))
        try:
            yatai_client = YataiClient()
            result = yatai_client.deployment.apply(deployment_yaml)
            if result.status.status_code != status_pb2.Status.OK:
                _echo(
                    'Failed to deploy deployment {name}. '
                    '{error_code}:{error_message}'.format(
                        name=deployment_yaml.get('name'),
                        error_code=status_pb2.Status.Code.Name(
                            result.status.status_code
                        ),
                        error_message=result.status.error_message,
                    ),
                    CLI_COLOR_ERROR,
                )
            else:
                if wait:
                    result_state = get_state_after_await_action_complete(
                        yatai_client=yatai_client,
                        name=deployment_yaml.get('name'),
                        namespace=deployment_yaml.get('namespace'),
                        message='Deploying deployment',
                    )
                    if result_state.status.status_code != status_pb2.Status.OK:
                        error_code = status_pb2.Status.Code.Name(
                            result_state.status.status_code
                        )
                        error_message = result_state.status.error_message
                        _echo(
                            f'Created deployment {deployment_yaml.get("name")}, '
                            f'failed to retrieve latest status. '
                            f'{error_code}:{error_message}'
                        )
                        return
                    result.deployment.state.CopyFrom(result_state.state)

                track_cli(
                    'deploy-deploy-success',
                    deployment_yaml.get('spec', {}).get('operator'),
                )
                _echo(
                    f'Successfully deploy spec to deployment '
                    f'{deployment_yaml.get("name")}',
                    CLI_COLOR_SUCCESS,
                )
                _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                'Failed to apply deployment {name}. Error message: {message}'.format(
                    name=deployment_yaml.get('name'), message=e
                )
            )

    @cli.command(help='Apply BentoService deployment from yaml file')
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
        try:
            yatai_client = YataiClient()
            result = yatai_client.deployment.apply(deployment_yaml)
            error_code, error_message = parse_pb_response_error_message(result.status)
            if error_code and error_message:
                if result.status.status_code != status_pb2.Status.OK:
                    _echo(
                        f'Failed to apply deployment {deployment_yaml.get("name")}. '
                        f'{error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
            else:
                if wait:
                    result_state = get_state_after_await_action_complete(
                        yatai_client=yatai_client,
                        name=deployment_yaml.get('name'),
                        namespace=deployment_yaml.get('namespace'),
                        message='Applying deployment',
                    )
                    error_code, error_message = parse_pb_response_error_message(
                        result_state.status
                    )
                    if error_code and error_message:
                        _echo(
                            f'Created deployment {deployment_yaml.get("name")}, '
                            f'failed to retrieve latest status. '
                            f'{error_code}:{error_message}',
                        )
                        return
                    result.deployment.state.CopyFrom(result_state.state)

                track_cli(
                    'deploy-apply-success',
                    deployment_yaml.get('spec', {}).get('operator'),
                )
                _echo(
                    'Successfully applied spec to deployment {}'.format(
                        deployment_yaml.get('name')
                    ),
                    CLI_COLOR_SUCCESS,
                )
                _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                'Failed to apply deployment {name}. Error message: {message}'.format(
                    name=deployment_yaml.get('name'), message=e
                )
            )

    @cli.command(help='Delete deployment')
    @click.option('--name', type=click.STRING, required=True, help='Deployment name')
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
        error_code, error_message = parse_pb_response_error_message(
            get_deployment_result.status
        )
        if error_code and error_message:
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
        error_code, error_message = parse_pb_response_error_message(result.status)
        if error_code and error_message:
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
