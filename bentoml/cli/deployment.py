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
    parse_bento_tag_callback,
    parse_yaml_file_callback,
)
from bentoml.proto.deployment_pb2 import DeploymentSpec, DeploymentState
from bentoml.proto.status_pb2 import Status
from bentoml.utils import pb_to_yaml
from bentoml.utils.usage_stats import track_cli
from bentoml.exceptions import BentoMLDeploymentException, BentoMLException
from bentoml.cli.utils import Spinner
from bentoml.yatai.python_api import (
    apply_deployment,
    create_deployment,
    delete_deployment,
    get_deployment,
    describe_deployment,
)

# pylint: disable=unused-variable

logger = logging.getLogger(__name__)


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
    name, namespace, message, timeout_limit=600, wait_time=5
):
    start_time = time.time()

    with Spinner(message):
        while (time.time() - start_time) < timeout_limit:
            result = describe_deployment(namespace, name)
            if result.state.state is DeploymentState.PENDING:
                time.sleep(wait_time)
                continue
            else:
                break
    return result


def get_deployment_sub_command():
    @click.group(
        help='Commands for creating and managing BentoService deployments on cloud'
        'computing platforms or kubernetes cluster'
    )
    def deployment():
        pass

    @deployment.command(
        short_help='Create a BentoService model serving deployment',
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @click.argument("name", type=click.STRING, required=True)
    @click.option(
        '-b',
        '--bento',
        type=click.STRING,
        required=True,
        callback=parse_bento_tag_callback,
        help='Target BentoService to be deployed, referenced by its name and version '
        'in format of name:version. For example: "iris_classifier:v1.2.0"',
    )
    @click.option(
        '-p',
        '--platform',
        type=click.Choice(
            ['aws-lambda', 'gcp-function', 'aws-sagemaker', 'kubernetes', 'custom'],
            case_sensitive=False,
        ),
        required=True,
        help='Which cloud platform to deploy this BentoService to',
    )
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "default" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        help='Key:value pairs that are attached to deployments and intended to be used'
        'to specify identifying attributes of the deployments that are meaningful to '
        'users',
    )
    @click.option(
        '--annotations',
        type=click.STRING,
        help='Used to attach arbitary metadata to BentoService deployments, BentoML '
        'library and other plugins can then retrieve this metadata.',
    )
    @click.option(
        '--region',
        help='Directly mapping to cloud provider region. Option applicable to platform:'
        'AWS Lambda, AWS SageMaker, GCP Function',
    )
    @click.option(
        '--instance-type',
        help='Type of instance will be used for inference. Option applicable to '
        'platform: AWS SageMaker, AWS Lambda, GCP Function',
    )
    @click.option(
        '--instance-count',
        help='Number of instance will be used. Option applicable to platform: AWS '
        'SageMaker',
        type=click.INT,
    )
    @click.option(
        '--api-name',
        help='User defined API function will be used for inference. Option applicable'
        'to platform: AWS SageMaker',
    )
    @click.option(
        '--kube-namespace',
        help='Namespace for kubernetes deployment. Option applicable to platform: '
        'Kubernetes',
    )
    @click.option(
        '--replicas',
        help='Number of replicas. Option applicable to platform: Kubernetes',
    )
    @click.option(
        '--service-name',
        help='Name for service. Option applicable to platform: Kubernetes',
    )
    @click.option(
        '--service-type', help='Service Type. Option applicable to platform: Kubernetes'
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for cloud resources to complete creation or until an error is '
        'encountered. When set to no-wait, CLI will return immediately after sending'
        'request to cloud platform.',
    )
    def create(
        name,
        bento,
        platform,
        output,
        namespace,
        labels,
        annotations,
        region,
        instance_type,
        instance_count,
        api_name,
        kube_namespace,
        replicas,
        service_name,
        service_type,
        wait,
    ):
        # converting platform parameter to DeploymentOperator name in proto
        # e.g. 'aws-lambda' to 'AWS_LAMBDA'
        platform = platform.replace('-', '_').upper()
        operator = DeploymentSpec.DeploymentOperator.Value(platform)

        track_cli('deploy-create', platform)
        bento_name, bento_version = bento.split(':')
        operator_spec = {
            'region': region,
            'instanceType': instance_type,
            'instanceCount': instance_count,
            'apiName': api_name,
            'kubeNamespace': kube_namespace,
            'replicas': replicas,
            'serviceName': service_name,
            'serviceType': service_type,
        }
        try:
            result = create_deployment(
                name,
                namespace,
                bento_name,
                bento_version,
                operator,
                operator_spec,
                parse_key_value_pairs(labels),
                parse_key_value_pairs(annotations),
            )

            if result.status.status_code != Status.OK:
                _echo(
                    'Failed to create deployment {name}. {error_code}: '
                    '{error_message}'.format(
                        name=name,
                        error_code=Status.Code.Name(result.status.status_code),
                        error_message=result.status.error_message,
                    ),
                    CLI_COLOR_ERROR,
                )
            else:
                if wait:
                    result_state = get_state_after_await_action_complete(
                        name=name, namespace=namespace, message='Creating deployment '
                    )
                    result.deployment.state.CopyFrom(result_state.state)

                _echo(
                    'Successfully created deployment {}'.format(name), CLI_COLOR_SUCCESS
                )
                _print_deployment_info(result.deployment, output)
        except BentoMLDeploymentException as error:
            _echo(
                'Failed to create deployment {name}. {error_message}'.format(
                    name=name, error_message=str(error)
                ),
                CLI_COLOR_ERROR,
            )

    @deployment.command(help='Apply model service deployment from yaml file')
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
            result = apply_deployment(deployment_yaml)
            if result.status.status_code != Status.OK:
                _echo(
                    'Failed to apply deployment {name}. code: {error_code}, message: '
                    '{error_message}'.format(
                        name=deployment_yaml.get('name'),
                        error_code=Status.Code.Name(result.status.status_code),
                        error_message=result.status.error_message,
                    ),
                    CLI_COLOR_ERROR,
                )
            else:
                if wait:
                    result_state = get_state_after_await_action_complete(
                        name=deployment_yaml.get('name'),
                        namespace=deployment_yaml.get('namespace'),
                        message='Applying deployment',
                    )
                    result.deployment.state.CopyFrom(result_state.state)

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

    @deployment.command(help='Delete deployment')
    @click.argument("name", type=click.STRING, required=True)
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
        track_cli('deploy-delete')

        result = delete_deployment(name, namespace, force)
        if result.status.status_code == Status.OK:
            _echo(
                'Successfully deleted deployment "{}"'.format(name), CLI_COLOR_SUCCESS
            )
        else:
            _echo(
                'Failed to delete deployment {name}. code: {error_code}, message: '
                '{error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )

    @deployment.command(help='Get deployment current state')
    @click.argument("name", type=click.STRING, required=True)
    @click.option('-n', '--namespace', type=click.STRING, help='Deployment namespace')
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    def get(name, output, namespace):
        track_cli('deploy-get')

        result = get_deployment(namespace, name)
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to get deployment {name}. code: {error_code}, message: '
                '{error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            _print_deployment_info(result.deployment, output)

    @deployment.command(help='View the detailed state of the deployment')
    @click.argument("name", type=click.STRING, required=True)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "default" which'
        'can be changed in BentoML configuration file',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    def describe(name, output, namespace):
        track_cli('deploy-describe')

        result = describe_deployment(namespace, name)
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to describe deployment {name}. code: {error_code}, message: '
                '{error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            get_result = get_deployment(namespace, name)
            deployment_pb = get_result.deployment
            deployment_pb.state.CopyFrom(result.state)
            _print_deployment_info(deployment_pb, output)

    @deployment.command(name="list", help='List active deployments')
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "default" which'
        'can be changed in BentoML configuration file',
    )
    @click.option('--all-namespaces', is_flag=True)
    @click.option(
        '--limit', type=click.INT, help='Limit how many deployments will be retrieved'
    )
    @click.option(
        '--filters',
        type=click.STRING,
        help='List deployments containing the filter string in name or version',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        help='List deployments matching the giving labels',
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table']), default='table'
    )
    def list_deployments(output, limit, filters, labels, namespace, all_namespaces):
        track_cli('deploy-list')

        result = list_deployments(
            limit, filters, parse_key_value_pairs(labels), namespace, all_namespaces
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to list deployments. code: {error_code}, message: '
                '{error_message}'.format(
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            _print_deployments_info(result.deployments, output)

    return deployment
