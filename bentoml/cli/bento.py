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
import docker
from google.protobuf.json_format import MessageToJson
from tabulate import tabulate

from bentoml.cli.click_utils import (
    CLI_COLOR_WARNING,
    CLI_COLOR_SUCCESS,
    _echo,
    parse_bento_tag_callback,
    parse_bento_tag_list_callback,
)
from bentoml.cli.utils import (
    humanfriendly_age_from_datetime,
    _echo_docker_api_result,
    make_bento_name_docker_compatible,
    Spinner,
)
from bentoml.yatai.proto import status_pb2
from bentoml.utils import pb_to_yaml, status_pb_to_error_code_and_message
from bentoml.yatai.client import YataiClient
from bentoml.saved_bundle import safe_retrieve
from bentoml.exceptions import CLIException, BentoMLException


def _print_bento_info(bento, output_type):
    if output_type == 'yaml':
        _echo(pb_to_yaml(bento))
    else:
        _echo(MessageToJson(bento))


def _print_bento_table(bentos, wide=False):
    table = []
    if wide:
        headers = ['BENTO_SERVICE', 'CREATED_AT', 'APIS', 'ARTIFACTS', 'URI']
    else:
        headers = ['BENTO_SERVICE', 'AGE', 'APIS', 'ARTIFACTS']

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
            created_at = humanfriendly_age_from_datetime(
                bento.bento_service_metadata.created_at.ToDatetime()
            )
        row = [
            f'{bento.name}:{bento.version}',
            created_at,
            ', '.join(apis),
            ', '.join(artifacts),
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
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table', 'wide'])
    )
    def get(bento, limit, ascending_order, print_location, output):
        if ':' in bento:
            name, version = bento.split(':')
        else:
            name = bento
            version = None
        yatai_client = YataiClient()

        if name and version:
            output = output or 'json'
            get_bento_result = yatai_client.repository.get(name, version)
            if get_bento_result.status.status_code != status_pb2.Status.OK:
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
                bento_name=name, limit=limit, ascending_order=ascending_order
            )
            if list_bento_versions_result.status.status_code != status_pb2.Status.OK:
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
        yatai_client = YataiClient()
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
            if result.status.status_code != status_pb2.Status.OK:
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

        yatai_client = YataiClient()

        get_bento_result = yatai_client.repository.get(name, version)
        if get_bento_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_bento_result.status
            )
            raise CLIException(
                f'BentoService {name}:{version} not found - '
                f'{error_code}:{error_message}'
            )

        if get_bento_result.bento.uri.s3_presigned_url:
            bento_service_bundle_path = get_bento_result.bento.uri.s3_presigned_url
        else:
            bento_service_bundle_path = get_bento_result.bento.uri.uri

        safe_retrieve(bento_service_bundle_path, target_dir)

        click.echo('Service %s artifact directory => %s' % (name, target_dir))

    @cli.command(
        help='Containerize given Bento into a Docker image',
        short_help="Containerize given Bento into a Docker image",
    )
    @click.argument(
        "bento", type=click.STRING, callback=parse_bento_tag_callback,
    )
    @click.option('--push', is_flag=True)
    @click.option(
        '--docker-repository',
        help="Prepends specified Docker repository to image name.",
    )
    @click.option(
        '-u', '--username', type=click.STRING, required=False,
    )
    @click.option(
        '-p', '--password', type=click.STRING, required=False,
    )
    def containerize(bento, push, docker_repository, username, password):
        """Containerize specified BentoService.

        BENTO is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: "iris_classifier:v1.2.0"

        `bentoml containerize` command also supports the use of the `latest` tag
        and will automatically use the last built version of your Bento.

        You can also optionally provide a `--push` flag, which will push the built
        image to the Docker repository specified by the `--docker-repository`.

        If you would like to push to Docker Hub, `--docker-repository` can just be
        your Docker Hub username. Otherwise, the tag should include the hostname as
        well. For example, for Google Container Registry, the `--docker-repository`
        would be `[HOSTNAME]/[PROJECT-ID]`
        """
        name, version = bento.split(':')
        yatai_client = YataiClient()

        get_bento_result = yatai_client.repository.get(name, version)
        if get_bento_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_bento_result.status
            )
            raise CLIException(
                f'BentoService {name}:{version} not found - '
                f'{error_code}:{error_message}',
            )

        if get_bento_result.bento.uri.s3_presigned_url:
            bento_service_bundle_path = get_bento_result.bento.uri.s3_presigned_url
        else:
            bento_service_bundle_path = get_bento_result.bento.uri.uri

        _echo(f"Found Bento: {bento_service_bundle_path}")

        if docker_repository is not None:
            name = f'{docker_repository}/{name}'

        name, version = make_bento_name_docker_compatible(name, version)
        tag = f"{name}:{version}"
        if tag != bento:
            _echo(
                f'Bento name or tag was changed to be Docker compatible. \n'
                f'"{bento}" -> "{tag}"',
                CLI_COLOR_WARNING,
            )

        docker_api = docker.APIClient()
        try:
            with Spinner(f"Building Docker image: {name}\n"):
                _echo_docker_api_result(
                    docker_api.build(
                        path=bento_service_bundle_path, tag=tag, decode=True,
                    )
                )
        except docker.errors.APIError as error:
            raise CLIException(f'Could not build Docker image: {error}')

        _echo(
            f'Finished building {tag} from {bento}', CLI_COLOR_SUCCESS,
        )

        if push:
            if not docker_repository:
                raise CLIException('Docker Registry must be specified when pushing.')

            auth_config_payload = (
                {"username": username, "password": password}
                if username or password
                else None
            )

            try:
                with Spinner(f"Pushing docker image to {tag}\n"):
                    _echo_docker_api_result(
                        docker_api.push(
                            repository=name,
                            tag=version,
                            stream=True,
                            decode=True,
                            auth_config=auth_config_payload,
                        )
                    )
                _echo(
                    f'Pushed {tag} to {name}', CLI_COLOR_SUCCESS,
                )
            except (docker.errors.APIError, BentoMLException) as error:
                raise CLIException(f'Could not push Docker image: {error}')
