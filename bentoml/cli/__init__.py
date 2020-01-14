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

import re
import os
import json
import click
import logging
import tempfile
import subprocess
from pathlib import Path

from google.protobuf.json_format import MessageToJson
from ruamel.yaml import YAML
from tabulate import tabulate

from bentoml.bundler import (
    load,
    load_bento_service_api,
    load_saved_bundle_config,
    load_bento_service_metadata,
)
from bentoml.cli.aws_sagemaker import get_aws_sagemaker_sub_command
from bentoml.cli.utils import parse_pb_response_error_message
from bentoml.server import BentoAPIServer, get_docs
from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    conditional_argument,
    _echo,
    CLI_COLOR_ERROR,
)
from bentoml.cli.deployment import (
    add_additional_deployment_commands,
    _print_deployment_info,
    _print_deployments_info,
)
from bentoml.cli.config import get_configuration_sub_command
from bentoml.utils import ProtoMessageToDict, pb_to_yaml
from bentoml.utils.log import configure_logging
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient


def escape_shell_params(param):
    k, v = param.split('=')
    v = re.sub(r'([^a-zA-Z0-9])', r'\\\1', v)
    return '{}={}'.format(k, v)


def create_bento_service_cli(bundle_path=None):
    # pylint: disable=unused-variable

    @click.group(cls=BentoMLCommandGroup)
    @click.option(
        '-q',
        '--quiet',
        is_flag=True,
        default=False,
        help="Hide process logs and only print command results",
    )
    @click.option(
        '--verbose',
        '--debug',
        is_flag=True,
        default=False,
        help="Print verbose debugging information for BentoML developer",
    )
    @click.version_option()
    @click.pass_context
    def bentoml_cli(ctx, verbose, quiet):
        """
        BentoML CLI tool
        """
        ctx.verbose = verbose
        ctx.quiet = quiet

        if verbose:
            from bentoml import config

            config().set('core', 'debug', 'true')
            configure_logging(logging.DEBUG)
        elif quiet:
            configure_logging(logging.ERROR)
        else:
            configure_logging()  # use default setting in local bentoml.cfg

    # Example Usage: bentoml {API_NAME} {BUNDLE_PATH} --input=...
    @bentoml_cli.command(
        default_command=True,
        default_command_usage="{API_NAME} {BUNDLE_PATH} --input=...",
        default_command_display_name="<API_NAME>",
        short_help="Run API function",
        help="Run a API defined in saved BentoService bundle from command line",
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @click.argument("api-name", type=click.STRING)
    @conditional_argument(bundle_path is None, "bundle-path", type=click.STRING)
    @click.option(
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    @click.pass_context
    def run(ctx, api_name, bundle_path=bundle_path, with_conda=False):
        if with_conda:
            config = load_saved_bundle_config(bundle_path)
            metadata = config['metadata']
            env_name = metadata['service_name'] + '_' + metadata['service_version']

            yaml = YAML()
            yaml.default_flow_style = False
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            env_path = tmpf.name
            yaml.dump(config['env']['conda_env'], Path(env_path))

            pip_req = os.path.join(bundle_path, 'requirements.txt')

            subprocess.call(
                'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
                'parameter requires conda but it\'s not installed."; exit 1; }} && '
                'conda env update -n {env_name} -f {env_file} && '
                'conda init bash && '
                'eval "$(conda shell.bash hook)" && '
                'conda activate {env_name} && '
                '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
                'dependencies."; }} &&'
                'bentoml {api_name} {bundle_path} {args}'.format(
                    env_name=env_name,
                    env_file=env_path,
                    bundle_path=bundle_path,
                    api_name=api_name,
                    args=' '.join(map(escape_shell_params, ctx.args)),
                    pip_req=pip_req,
                ),
                shell=True,
            )
            return

        track_cli('run')

        api = load_bento_service_api(bundle_path, api_name)
        api.handle_cli(ctx.args)

    # Example Usage: bentoml info {BUNDLE_PATH}
    @bentoml_cli.command(
        help="List all APIs defined in the BentoService loaded from saved bundle",
        short_help="List APIs",
    )
    @conditional_argument(bundle_path is None, "bundle-path", type=click.STRING)
    def info(bundle_path=bundle_path):
        """
        List all APIs defined in the BentoService loaded from saved bundle
        """
        track_cli('info')
        bento_service_metadata_pb = load_bento_service_metadata(bundle_path)
        output = json.dumps(ProtoMessageToDict(bento_service_metadata_pb), indent=2)
        _echo(output)

    # Example usage: bentoml open-api-spec {BUNDLE_PATH}
    @bentoml_cli.command(
        name="open-api-spec",
        help="Display API specification JSON in Open-API format",
        short_help="Display OpenAPI/Swagger JSON specs",
    )
    @conditional_argument(bundle_path is None, "bundle-path", type=click.STRING)
    def open_api_spec(bundle_path=bundle_path):
        track_cli('open-api-spec')
        bento_service = load(bundle_path)

        _echo(json.dumps(get_docs(bento_service), indent=2))

    # Example Usage: bentoml serve {BUNDLE_PATH} --port={PORT}
    @bentoml_cli.command(
        help="Start REST API server hosting BentoService loaded from saved bundle",
        short_help="Start local rest server",
    )
    @conditional_argument(bundle_path is None, "bundle-path", type=click.STRING)
    @click.option(
        "--port",
        type=click.INT,
        default=BentoAPIServer._DEFAULT_PORT,
        help="The port to listen on for the REST api server, default is 5000.",
    )
    @click.option(
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    def serve(port, bundle_path=bundle_path, with_conda=False):
        if with_conda:
            config = load_saved_bundle_config(bundle_path)
            metadata = config['metadata']
            env_name = metadata['service_name'] + '_' + metadata['service_version']
            pip_req = os.path.join(bundle_path, 'requirements.txt')

            subprocess.call(
                'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
                'parameter requires conda but it\'s not installed."; exit 1; }} && '
                'conda env update -n {env_name} -f {env_file} && '
                'conda init bash && '
                'eval "$(conda shell.bash hook)" && '
                'conda activate {env_name} && '
                '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
                'dependencies."; }} &&'
                'bentoml serve {bundle_path} --port {port}'.format(
                    env_name=env_name,
                    env_file=os.path.join(bundle_path, 'environment.yml'),
                    bundle_path=bundle_path,
                    port=port,
                    pip_req=pip_req,
                ),
                shell=True,
            )
            return

        track_cli('serve')

        bento_service = load(bundle_path)
        server = BentoAPIServer(bento_service, port=port)
        server.start()

    # Example Usage:
    # bentoml serve-gunicorn {BUNDLE_PATH} --port={PORT} --workers={WORKERS}
    @bentoml_cli.command(
        help="Start REST API server from saved BentoService bundle with gunicorn",
        short_help="Start local gunicorn server",
    )
    @conditional_argument(bundle_path is None, "bundle-path", type=click.STRING)
    @click.option("-p", "--port", type=click.INT, default=None)
    @click.option(
        "-w",
        "--workers",
        type=click.INT,
        default=None,
        help="Number of workers will start for the gunicorn server",
    )
    @click.option("--timeout", type=click.INT, default=None)
    @click.option(
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    def serve_gunicorn(
        port, workers, timeout, bundle_path=bundle_path, with_conda=False
    ):
        if with_conda:
            config = load_saved_bundle_config(bundle_path)
            metadata = config['metadata']
            env_name = metadata['service_name'] + '_' + metadata['service_version']
            pip_req = os.path.join(bundle_path, 'requirements.txt')

            subprocess.call(
                'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
                'parameter requires conda but it\'s not installed."; exit 1; }} && '
                'conda env update -n {env_name} -f {env_file} && '
                'conda init bash && '
                'eval "$(conda shell.bash hook)" && '
                'conda activate {env_name} && '
                '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
                'dependencies."; }} &&'
                'bentoml serve_gunicorn {bundle_path} -p {port} -w {workers} '
                '--timeout {timeout}'.format(
                    env_name=env_name,
                    env_file=os.path.join(bundle_path, 'environment.yml'),
                    bundle_path=bundle_path,
                    port=port,
                    workers=workers,
                    timeout=timeout,
                    pip_req=pip_req,
                ),
                shell=True,
            )
            return

        track_cli('serve_gunicorn')

        from bentoml.server.gunicorn_server import GunicornBentoServer

        gunicorn_app = GunicornBentoServer(bundle_path, port, workers, timeout)
        gunicorn_app.run()

    # pylint: enable=unused-variable
    return bentoml_cli


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


def create_bentoml_cli():
    _cli = create_bento_service_cli()

    # Commands created here aren't mean to be used from generated BentoService CLI when
    # installed as PyPI package. The are only used as part of BentoML cli command.

    @_cli.command(help='Get BentoML resources')
    @click.argument(
        'resource',
        type=click.Choice(['deployment', 'bento']),
        default='deployment',
        required=True,
    )
    @click.option('-n', '--deployment-name', type=click.STRING, help='Deployment name')
    @click.option('-n', '--bento-name', type=click.STRING, help='BentoService name')
    @click.option(
        '-n', '--bento-version', type=click.STRING, help='BentoService version'
    )
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
    def get(
        resource,
        deployment_name,
        namespace,
        bento_name,
        bento_version,
        all_namespaces,
        limit,
        filters,
        labels,
        output,
    ):
        yatai_client = YataiClient()
        if resource == 'deployment':
            if deployment_name:
                track_cli('deploy-get')
                get_result = yatai_client.deployment.get(namespace, deployment_name)
                error_code, error_message = parse_pb_response_error_message(
                    get_result.status
                )
                if error_code and error_message:
                    _echo(
                        f'Failed to get deployment {deployment_name}. '
                        f'{error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
                    return
                describe_result = yatai_client.deployment.describe(
                    namespace, deployment_name
                )
                error_code, error_message = parse_pb_response_error_message(
                    describe_result.status
                )
                if error_code and error_message:
                    _echo(
                        f'Failed to retrieve the latest status for Sagemaker deployment'
                        f' {deployment_name}. {error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
                    return
                get_result.deployment.state.CopyFrom(describe_result.state)
                _print_deployment_info(get_result.deployment, output)
                return
            else:
                track_cli('deploy-list')
                list_result = yatai_client.deployment.list(
                    limit, filters, labels, namespace, all_namespaces
                )
                error_code, error_message = parse_pb_response_error_message(
                    list_result.status
                )
                if error_code and error_message:
                    _echo(
                        f'Failed to list deployments. ' f'{error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
                else:
                    _print_deployments_info(list_result.deployments, output)
        else:
            if bento_name and bento_version:
                track_cli('bento-get')
                output = output or 'yaml'
                get_bento_result = yatai_client.repository.get(
                    bento_name, bento_version
                )
                error_code, error_message = parse_pb_response_error_message(
                    get_bento_result.status
                )
                if error_code and error_message:
                    _echo(
                        f'Failed to get BentoService{bento_name}:{bento_version} '
                        f'{error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
                    return
                _print_bento_info(get_bento_result.bento, output)
                return
            elif bento_name:
                track_cli('bento-list')
                output = output or 'table'
                list_bento_versions_result = yatai_client.repository.list(
                    bento_name=bento_name, filters=filters, limit=limit
                )
                error_code, error_message = parse_pb_response_error_message(
                    list_bento_versions_result.status
                )
                if error_code and error_message:
                    _echo(
                        f'Failed to list versions for BentoService {bento_name} '
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
                error_code, error_message = parse_pb_response_error_message(
                    list_bentos_result.status
                )
                if error_code and error_message:
                    _echo(
                        f'Failed to list BentoServices '
                        f'{error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
                    return

                _print_bentos_info(list_bentos_result.bentos, output)
                return

    config_sub_command = get_configuration_sub_command()
    aws_sagemaker_sub_command = get_aws_sagemaker_sub_command()
    _cli.add_command(config_sub_command)
    _cli.add_command(aws_sagemaker_sub_command)

    add_additional_deployment_commands(_cli)

    return _cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
