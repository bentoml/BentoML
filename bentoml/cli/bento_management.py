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
import os
from tabulate import tabulate

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.cli.click_utils import (
    _echo,
    parse_bento_tag_list_callback,
)
from bentoml.cli.utils import (
    human_friendly_age_from_datetime,
    get_default_yatai_client,
    _format_labels_for_print,
)
from bentoml.utils import pb_to_yaml, status_pb_to_error_code_and_message
from bentoml.saved_bundle import safe_retrieve
from bentoml.exceptions import CLIException


yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')


def _print_bento_info(bento, output_type):
    if output_type == 'yaml':
        _echo(pb_to_yaml(bento))
    else:
        from google.protobuf.json_format import MessageToJson

        _echo(MessageToJson(bento))


def _print_bento_table(bentos, wide=False):
    table = []
    if wide:
        headers = ['BENTO_SERVICE', 'CREATED_AT', 'APIS', 'ARTIFACTS', 'LABELS', 'URI']
    else:
        headers = ['BENTO_SERVICE', 'AGE', 'APIS', 'ARTIFACTS', 'LABELS']

    for bento in bentos:
        artifacts = [
            f'{artifact.name}<{artifact.artifact_type}>'
            for artifact in bento.bento_service_metadata.artifacts
        ]
        apis = [
            f'{api.name}<{api.input_type}:{api.output_type}>'
            for api in bento.bento_service_metadata.apis
        ]
        if wide:
            created_at = bento.bento_service_metadata.created_at.ToDatetime().strftime(
                "%Y-%m-%d %H:%M"
            )
        else:
            created_at = human_friendly_age_from_datetime(
                bento.bento_service_metadata.created_at.ToDatetime()
            )
        row = [
            f'{bento.name}:{bento.version}',
            created_at,
            ', '.join(apis),
            ', '.join(artifacts),
            _format_labels_for_print(bento.bento_service_metadata.labels),
        ]
        if wide:
            row.append(bento.uri.uri)
        table.append(row)

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
    @click.option('--print-location', is_flag=True)
    @click.option(
        '--labels',
        type=click.STRING,
        help="Label query to filter BentoServices, supports '=', '!=', 'IN', 'NotIn', "
        "'Exists', and 'DoesNotExist'. (e.g. key1=value1, key2!=value2, key3 "
        "In (value3, value3a), key4 DoesNotExist)",
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table', 'wide'])
    )
    def get(bento, limit, ascending_order, print_location, labels, output):
        if ':' in bento:
            name, version = bento.split(':')
        else:
            name = bento
            version = None
        yatai_client = get_default_yatai_client()

        if name and version:
            output = output or 'json'
            get_bento_result = yatai_client.repository.get(name, version)
            if get_bento_result.status.status_code != yatai_proto.status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    get_bento_result.status
                )
                raise CLIException(f'{error_code}:{error_message}')
            if print_location:
                _echo(get_bento_result.bento.uri.uri)
                return
            _print_bento_info(get_bento_result.bento, output)
        elif name:
            output = output or 'table'
            list_bento_versions_result = yatai_client.repository.list(
                bento_name=name,
                limit=limit,
                labels=labels,
                ascending_order=ascending_order,
            )
            if (
                list_bento_versions_result.status.status_code
                != yatai_proto.status_pb2.Status.OK
            ):
                error_code, error_message = status_pb_to_error_code_and_message(
                    list_bento_versions_result.status
                )
                raise CLIException(f'{error_code}:{error_message}')

            _print_bentos_info(list_bento_versions_result.bentos, output)

    @cli.command(name='list', help='List BentoServices information')
    @click.option(
        '--limit', type=click.INT, help='Limit how many BentoServices will be retrieved'
    )
    @click.option(
        '--offset', type=click.INT, help='How many BentoServices will be skipped'
    )
    @click.option(
        '--labels',
        type=click.STRING,
        help="Label query to filter BentoServices, supports '=', '!=', 'IN', 'NotIn', "
        "'Exists', and 'DoesNotExist'. (e.g. key1=value1, key2!=value2, key3 "
        "In (value3, value3a), key4 DoesNotExist)",
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
    def list_bentos(limit, offset, labels, order_by, ascending_order, output):
        yatai_client = get_default_yatai_client()
        list_bentos_result = yatai_client.repository.list(
            limit=limit,
            offset=offset,
            labels=labels,
            order_by=order_by,
            ascending_order=ascending_order,
        )
        if list_bentos_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                list_bentos_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')

        _print_bentos_info(list_bentos_result.bentos, output)

    @cli.command()
    @click.argument("bentos", type=click.STRING, callback=parse_bento_tag_list_callback)
    @click.option(
        '-y', '--yes', '--assume-yes', is_flag=True, help='Automatic yes to prompts'
    )
    def delete(bentos, yes):
        """Delete saved BentoService.

        BENTO is the target BentoService to be deleted, referenced by its name and
        version in format of name:version. For example: "iris_classifier:v1.2.0"

        `bentoml delete` command also supports deleting multiple saved BentoService at
        once, by providing name version tag separated by ",", for example:

        `bentoml delete iris_classifier:v1.2.0,my_svc:v1,my_svc2:v3`
        """
        yatai_client = get_default_yatai_client()
        for bento in bentos:
            name, version = bento.split(':')
            if not name and not version:
                raise CLIException(
                    'BentoService name or version is missing. Please provide in the '
                    'format of name:version'
                )
            if not yes and not click.confirm(
                f'Are you sure about delete {bento}? This will delete the BentoService '
                f'saved bundle files permanently'
            ):
                return
            result = yatai_client.repository.dangerously_delete_bento(
                name=name, version=version
            )
            if result.status.status_code != yatai_proto.status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                raise CLIException(f'{error_code}:{error_message}')
            _echo(f'BentoService {name}:{version} deleted')

    @cli.command(
        help='Retrieves BentoService artifacts into a target directory',
        short_help="Retrieves BentoService artifacts into a target directory",
    )
    @click.argument("bento", type=click.STRING)
    @click.option(
        '--target_dir',
        help="Directory to put artifacts into. Defaults to pwd.",
        default=os.getcwd(),
    )
    def retrieve(bento, target_dir):
        if ':' not in bento:
            _echo(f'BentoService {bento} invalid - specify name:version')
            return
        name, version = bento.split(':')

        yatai_client = get_default_yatai_client()

        get_bento_result = yatai_client.repository.get(name, version)
        if get_bento_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_bento_result.status
            )
            raise CLIException(
                f'Failed to access BentoService {name}:{version} - '
                f'{error_code}:{error_message}'
            )

        if get_bento_result.bento.uri.s3_presigned_url:
            bento_service_bundle_path = get_bento_result.bento.uri.s3_presigned_url
        if get_bento_result.bento.uri.gcs_presigned_url:
            bento_service_bundle_path = get_bento_result.bento.uri.gcs_presigned_url
        else:
            bento_service_bundle_path = get_bento_result.bento.uri.uri

        safe_retrieve(bento_service_bundle_path, target_dir)

        click.echo('Service %s artifact directory => %s' % (name, target_dir))
