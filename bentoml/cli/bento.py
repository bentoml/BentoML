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
from google.protobuf.json_format import MessageToJson
from tabulate import tabulate

from bentoml.cli.click_utils import (
    CLI_COLOR_ERROR,
    _echo,
)
from bentoml.proto import status_pb2
from bentoml.utils import pb_to_yaml, status_pb_to_error_code_and_message
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient


def _print_bento_info(bento, output_type):
    if output_type == 'yaml':
        _echo(pb_to_yaml(bento))
    else:
        _echo(MessageToJson(bento))


def _print_bento_table(bentos, wide=False):
    table = []
    headers = ['BENTO_SERVICE', 'CREATED_AT', 'APIS', 'ARTIFACTS']
    for bento in bentos:
        artifacts = [
            f'{artifact.name}::{artifact.artifact_type}'
            for artifact in bento.bento_service_metadata.artifacts
        ]
        apis = [
            f'{api.name}::{api.handler_type}'
            for api in bento.bento_service_metadata.apis
        ]
        row = [
            f'{bento.name}:{bento.version}',
            bento.bento_service_metadata.created_at.ToDatetime().strftime(
                "%Y-%m-%d %H:%M"
            ),
            ', '.join(apis),
            ', '.join(artifacts),
        ]
        table.append(row)

    if wide:
        headers.append('URI')
        for i, bento in enumerate(bentos):
            table[i].append(bento.uri.uri)

    table_display = tabulate(table, headers, tablefmt='plain')
    _echo(table_display)


def _print_bentos_info(bentos, output_type):
    if output_type == 'table':
        _print_bento_table(bentos)
    elif output_type == 'wide':
        _print_bento_table(bentos, wide=True)
    else:
        for bento in bentos:
            _print_bento_info(bento, output_type)


def add_bento_sub_command(cli):
    # pylint: disable=unused-variable
    @cli.command(help='Get BentoService information')
    @click.argument('bento', type=click.STRING)
    @click.option(
        '--limit', type=click.INT, help='Limit how many resources will be retrieved'
    )
    @click.option('--ascending-order', is_flag=True)
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table', 'wide'])
    )
    def get(bento, limit, ascending_order, output):
        if ':' in bento:
            name, version = bento.split(':')
        else:
            name = bento
            version = None
        yatai_client = YataiClient()

        if name and version:
            track_cli('bento-get')
            output = output or 'json'
            get_bento_result = yatai_client.repository.get(name, version)
            if get_bento_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    get_bento_result.status
                )
                _echo(
                    f'BentoService {name}:{version} not found - '
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
                bento_name=name, limit=limit, ascending_order=ascending_order
            )
            if list_bento_versions_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    list_bento_versions_result.status
                )
                _echo(
                    f'Failed to list versions for BentoService {name} '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return

            _print_bentos_info(list_bento_versions_result.bentos, output)

    @cli.command(name='list', help='List BentoServices information')
    @click.option(
        '--limit', type=click.INT, help='Limit how many BentoServices will be retrieved'
    )
    @click.option(
        '--offset', type=click.INT, help='How many BentoServices will be skipped'
    )
    @click.option(
        '--order-by', type=click.Choice(['created_at', 'name']), default='created_at',
    )
    @click.option('--ascending-order', is_flag=True)
    @click.option(
        '-o',
        '--output',
        type=click.Choice(['json', 'yaml', 'table', 'wide']),
        default='table',
    )
    def list_bentos(limit, offset, order_by, ascending_order, output):
        yatai_client = YataiClient()
        track_cli('bento-list')
        list_bentos_result = yatai_client.repository.list(
            limit=limit,
            offset=offset,
            order_by=order_by,
            ascending_order=ascending_order,
        )
        if list_bentos_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                list_bentos_result.status
            )
            _echo(
                f'Failed to list BentoServices ' f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return

        _print_bentos_info(list_bentos_result.bentos, output)

    @cli.command(help='Delete BentoService')
    @click.argument('bento', type=click.STRING)
    @click.option(
        '-y', '--yes', '--assume-yes', is_flag=True, help='Automatic yes to prompts'
    )
    def delete(bento, yes):
        yatai_client = YataiClient()
        name, version = bento.split(':')
        if not name and not version:
            _echo(
                'BentoService name or version is missing. Please provide in the '
                'format of name:version',
                CLI_COLOR_ERROR,
            )
            return
        if not yes and not click.confirm(
            f'Are you sure about delete {bento}? This will delete the BentoService '
            f'saved bundle files permanently'
        ):
            return
        result = yatai_client.repository.dangerously_delete_bento(
            name=name, version=version
        )
        if result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            _echo(
                f'Failed to delete Bento {name}:{version} '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        _echo(f'BentoService {name}:{version} deleted')
