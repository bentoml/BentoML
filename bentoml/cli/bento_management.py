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
import os
import click
from tabulate import tabulate

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.cli.click_utils import (
    _echo,
    parse_bento_tag_list_callback,
)
from bentoml.cli.utils import (
    human_friendly_age_from_datetime,
    _format_labels_for_print,
)
from bentoml.utils import pb_to_yaml
from bentoml.yatai.client import get_yatai_client

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
        '--yatai-url',
        type=click.STRING,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table', 'wide'])
    )
    def get(bento, limit, ascending_order, print_location, labels, yatai_url, output):
        yc = get_yatai_client(yatai_url)
        if ':' in bento:
            result = yc.repository.get(bento)
            if print_location:
                _echo(result.uri.uri)
            else:
                _print_bento_info(result, output)
        else:
            output = output or 'table'
            result = yc.repository.list(
                bento_name=bento,
                limit=limit,
                ascending_order=ascending_order,
                labels=labels,
            )
            _print_bentos_info(result, output)

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
        '--yatai-url',
        type=click.STRING,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    @click.option(
        '-o',
        '--output',
        type=click.Choice(['json', 'yaml', 'table', 'wide']),
        default='table',
    )
    def list_bentos(
        limit, offset, labels, order_by, ascending_order, yatai_url, output
    ):
        yc = get_yatai_client(yatai_url)
        result = yc.repository.list(
            limit=limit,
            offset=offset,
            labels=labels,
            order_by=order_by,
            ascending_order=ascending_order,
        )
        _print_bentos_info(result, output)

    @cli.command(
        help='Delete bento. To delete multiple bentos provide the name '
        'version tag separated by "," for example "bentoml delete name:v1,name:v2',
    )
    @click.argument("bentos", type=click.STRING, callback=parse_bento_tag_list_callback)
    @click.option(
        '--yatai-url',
        type=click.STRING,
        help='Remote YataiService URL. Optional. Example: '
        '"--yatai-url http://localhost:50050"',
    )
    @click.option(
        '-y', '--yes', '--assume-yes', is_flag=True, help='Automatic yes to prompts'
    )
    def delete(bentos, yatai_url, yes):
        """Delete saved BentoService.

        BENTO is the target BentoService to be deleted, referenced by its name and
        version in format of name:version. For example: "iris_classifier:v1.2.0"

        `bentoml delete` command also supports deleting multiple saved BentoService at
        once, by providing name version tag separated by ",", for example:

        `bentoml delete iris_classifier:v1.2.0,my_svc:v1,my_svc2:v3`
        """
        for bento in bentos:
            if not yes and not click.confirm(
                f'Are you sure about delete {bento}? This will delete the BentoService '
                f'saved bundle files permanently'
            ):
                return
            yc = get_yatai_client(yatai_url)
            yc.repository.delete(bento)
            _echo(f'BentoService {bento} deleted')

    @cli.command(help='Pull BentoService from remote yatai server',)
    @click.argument("bento", type=click.STRING)
    @click.option(
        '--yatai-url',
        required=True,
        help='Remote YataiService URL. Example: "--yatai-url http://localhost:50050"',
    )
    def pull(bento, yatai_url):
        if ':' not in bento:
            _echo(f'BentoService {bento} invalid - specify name:version')
            return
        yc = get_yatai_client(yatai_url)
        yc.repository.pull(bento=bento)
        _echo(f'Pulled {bento} from {yatai_url}')

    @cli.command(help='Push BentoService to remote yatai server')
    @click.argument("bento", type=click.STRING)
    @click.option(
        '--yatai-url',
        required=True,
        help='Remote YataiService URL. Example: "--yatai-url http://localhost:50050"',
    )
    def push(bento, yatai_url):
        if ':' not in bento:
            _echo(f'BentoService {bento} invalid - specify name:version')
            return
        yc = get_yatai_client(yatai_url)
        yc.repository.push(bento=bento)
        _echo(f'Pushed {bento} to {yatai_url}')

    @cli.command(help='Retrieve')
    @click.argument("bento", type=click.STRING)
    @click.option(
        '--yatai-url',
        help='Remote YataiService URL. Example: "--yatai-url http://localhost:50050"',
    )
    @click.option(
        '--target_dir',
        help="Target directory to save BentoService. Defaults to the current directory",
        default=os.getcwd(),
    )
    def retrieve(bento, yatai_url, target_dir):
        yc = get_yatai_client(yatai_url)
        bento_pb = yc.repository.get(bento)
        yc.repository.download_to_directory(bento_pb, target_dir)

        _echo(f'Save {bento} artifact to directory {target_dir}')
