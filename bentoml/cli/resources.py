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
import json

import click
from google.protobuf.json_format import MessageToJson
from tabulate import tabulate

from bentoml.cli.utils import parse_pb_response_error_message
from bentoml.cli.click_utils import (
    CLI_COLOR_ERROR,
    _echo,
)
from bentoml.cli.deployment import (
    _print_deployment_info,
    _print_deployments_info,
)
from bentoml.proto import status_pb2
from bentoml.utils import pb_to_yaml
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient


def _print_bento_info(bento, output_type):
    if output_type == 'yaml':
        result = pb_to_yaml(bento)
    else:
        result = MessageToJson(bento)
        _echo(json.dumps(result, indent=2, separators=(',', ': ')))
        return
    _echo(result)


def _print_bento_table(bentos):
    table = []
    headers = ['NAME', 'VERSION', 'CREATED_AT', 'ARTIFACTS', 'HANDLERS']
    for bento in bentos:
        artifacts = [
            artifact.artifact_type
            for artifact in bento.bento_service_metadata.artifacts
        ]
        handlers = [api.handler_type for api in bento.bento_service_metadata.apis]
        row = [
            bento.name,
            bento.version,
            bento.bento_service_metadata.created_at.ToDatetime(),
            ', '.join(artifacts),
            ', '.join(handlers),
        ]
        table.append(row)
    table_display = tabulate(table, headers, tablefmt='plain')
    _echo(table_display)


def _print_bentos_info(bentos, output_type):
    if output_type == 'table':
        _print_bento_table(bentos)
    else:
        for bento in bentos:
            _print_bento_info(bento, output_type)


def get_resources_sub_command():
    # pylint: disable=unused-variable

    @click.group(name='get', help='Get BentoML resources from db')
    def get_resource():
        pass

    @get_resource.command(help='Get deployment information from db')
    @click.option('-n', '--name', type=click.STRING, help='Deployment name')
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option('--all-namespaces', is_flag=True)
    @click.option(
        '--limit', type=click.INT, help='Limit how many resources will be retrieved'
    )
    @click.option(
        '--filters',
        type=click.STRING,
        help='List resources containing the filter string in name',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        help='List deployments matching the giving labels',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml', 'table']))
    def deployment(name, namespace, all_namespaces, limit, filters, labels, output):
        yatai_client = YataiClient()
        if name:
            track_cli('deploy-get')
            output = output or 'json'
            get_result = yatai_client.deployment.get(namespace, name)
            if get_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    get_result.status
                )
                _echo(
                    f'Failed to get deployment {name}. '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            describe_result = yatai_client.deployment.describe(
                namespace=namespace, name=name
            )
            if describe_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    describe_result.status
                )
                _echo(
                    f'Failed to retrieve the latest status for Sagemaker deployment'
                    f' {name}. {error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            get_result.deployment.state.CopyFrom(describe_result.state)
            _print_deployment_info(get_result.deployment, output)
            return
        else:
            track_cli('deploy-list')
            output = output or 'table'
            list_result = yatai_client.deployment.list(
                limit=limit,
                filters=filters,
                labels=labels,
                namespace=namespace,
                is_all_namespaces=all_namespaces,
            )
        if list_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = parse_pb_response_error_message(
                list_result.status
            )
            _echo(
                f'Failed to list deployments. ' f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
        else:
            _print_deployments_info(list_result.deployments, output)

    @get_resource.command(help='Get BentoService information from db')
    @click.option('-n', '--name', type=click.STRING, help='BentoService name')
    @click.option('-v', '--version', type=click.STRING, help='BentoService version')
    @click.option(
        '--limit', type=click.INT, help='Limit how many resources will be retrieved'
    )
    @click.option(
        '--filters',
        type=click.STRING,
        help='List resources containing the filter string in name',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml', 'table']))
    def bento(name, version, limit, filters, output):
        yatai_client = YataiClient()

        if name and version:
            track_cli('bento-get')
            output = output or 'json'
            get_bento_result = yatai_client.repository.get(name, version)
            if get_bento_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    get_bento_result.status
                )
                _echo(
                    f'Failed to get BentoService{name}:{version} '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            _print_bento_info(get_bento_result.bento, output)
            return
        elif name:
            track_cli('bento-list')
            output = output or 'table'
            list_bento_versions_result = yatai_client.repository.list(
                bento_name=name, filters=filters, limit=limit
            )
            if list_bento_versions_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    list_bento_versions_result.status
                )
                _echo(
                    f'Failed to list versions for BentoService {name} '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return

            _print_bentos_info(list_bento_versions_result.bentos, output)
            return
        else:
            track_cli('bento-list')
            output = output or 'table'
            list_bentos_result = yatai_client.repository.list(
                limit=limit, filters=filters
            )
            if list_bentos_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    list_bentos_result.status
                )
                _echo(
                    f'Failed to list BentoServices ' f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return

            _print_bentos_info(list_bentos_result.bentos, output)
            return

    return get_resource
