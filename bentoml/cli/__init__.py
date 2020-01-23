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
import tempfile
import subprocess
from pathlib import Path

from ruamel.yaml import YAML

from bentoml.bundler import (
    load,
    load_bento_service_api,
    load_saved_bundle_config,
    load_bento_service_metadata,
)
from bentoml.cli.aws_lambda import get_aws_lambda_sub_command
from bentoml.cli.aws_sagemaker import get_aws_sagemaker_sub_command
from bentoml.cli.bento import add_bento_sub_command
from bentoml.server import BentoAPIServer, get_docs
from bentoml.cli.click_utils import BentoMLCommandGroup, conditional_argument, _echo
from bentoml.cli.deployment import get_deployment_sub_command
from bentoml.cli.config import get_configuration_sub_command
from bentoml.utils import ProtoMessageToDict
from bentoml.utils.usage_stats import track_cli
from bentoml.utils.s3 import is_s3_url
from bentoml.yatai.client import YataiClient
from bentoml.proto import status_pb2
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.exceptions import BentoMLException


def escape_shell_params(param):
    k, v = param.split('=')
    v = re.sub(r'([^a-zA-Z0-9])', r'\\\1', v)
    return '{}={}'.format(k, v)


def run_with_conda_env(bundle_path, command):
    config = load_saved_bundle_config(bundle_path)
    metadata = config['metadata']
    env_name = metadata['service_name'] + '_' + metadata['service_version']

    yaml = YAML()
    yaml.default_flow_style = False
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    env_path = tmpf.name + '.yaml'
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
        'dependencies."; }} && {cmd}'.format(
            env_name=env_name, env_file=env_path, pip_req=pip_req, cmd=command,
        ),
        shell=True,
    )
    return


def create_bento_service_cli(pip_installed_bundle_path=None):
    # pylint: disable=unused-variable

    @click.group(cls=BentoMLCommandGroup)
    @click.version_option()
    def bentoml_cli():
        """
        BentoML CLI tool
        """

    def resolve_bundle_path(bento, pip_installed_bundle_path):
        if pip_installed_bundle_path:
            assert (
                bento is None
            ), "pip installed BentoService commands should not have Bento argument"
            return pip_installed_bundle_path

        if os.path.isdir(bento) or is_s3_url(bento):
            # bundler already support loading local and s3 path
            return bento

        elif ":" in bento:
            # assuming passing in BentoService in the form of Name:Version tag
            yatai_client = YataiClient()
            name, version = bento.split(':')
            get_bento_result = yatai_client.repository.get(name, version)
            if get_bento_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    get_bento_result.status
                )
                raise BentoMLException(
                    f'BentoService {name}:{version} not found - '
                    f'{error_code}:{error_message}'
                )
            return get_bento_result.bento.uri.uri
        else:
            raise BentoMLException(
                f'BentoService "{bento}" not found - either specify the file path of '
                f'the BentoService saved bundle, or the BentoService id in the form of '
                f'"name:version"'
            )

    # Example Usage: bentoml run {API_NAME} {BUNDLE_PATH} --input=...
    @bentoml_cli.command(
        help="Run a API defined in saved BentoService bundle from command line",
        short_help="Run API function",
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.argument("api_name", type=click.STRING)
    @click.argument('run_args', nargs=-1, type=click.UNPROCESSED)
    @click.option(
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    def run(api_name, run_args, bento=None, with_conda=False):
        track_cli('run')
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        if with_conda:
            run_with_conda_env(
                bento_service_bundle_path,
                'bentoml run {api_name} {bento} {args}'.format(
                    bento=bento_service_bundle_path,
                    api_name=api_name,
                    args=' '.join(map(escape_shell_params, run_args)),
                ),
            )
            return

        api = load_bento_service_api(bento_service_bundle_path, api_name)
        api.handle_cli(run_args)

    # Example Usage: bentoml info {BUNDLE_PATH}
    @bentoml_cli.command(
        help="List all APIs defined in the BentoService loaded from saved bundle",
        short_help="List APIs",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    def info(bento=None):
        """
        List all APIs defined in the BentoService loaded from saved bundle
        """
        track_cli('info')

        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        bento_service_metadata_pb = load_bento_service_metadata(
            bento_service_bundle_path
        )
        output = json.dumps(ProtoMessageToDict(bento_service_metadata_pb), indent=2)
        _echo(output)

    # Example usage: bentoml open-api-spec {BUNDLE_PATH}
    @bentoml_cli.command(
        name="open-api-spec",
        help="Display API specification JSON in Open-API format",
        short_help="Display OpenAPI/Swagger JSON specs",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    def open_api_spec(bento=None):
        track_cli('open-api-spec')

        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        bento_service = load(bento_service_bundle_path)

        _echo(json.dumps(get_docs(bento_service), indent=2))

    # Example Usage: bentoml serve {BUNDLE_PATH} --port={PORT}
    @bentoml_cli.command(
        help="Start REST API server hosting BentoService loaded from saved bundle",
        short_help="Start local rest server",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
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
    def serve(port, bento=None, with_conda=False):
        track_cli('serve')
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        if with_conda:
            run_with_conda_env(
                bento_service_bundle_path,
                'bentoml serve {bento} --port {port}'.format(
                    bento=bento_service_bundle_path, port=port,
                ),
            )
            return

        bento_service = load(bento_service_bundle_path)
        server = BentoAPIServer(bento_service, port=port)
        server.start()

    # Example Usage:
    # bentoml serve-gunicorn {BUNDLE_PATH} --port={PORT} --workers={WORKERS}
    @bentoml_cli.command(
        help="Start REST API server from saved BentoService bundle with gunicorn",
        short_help="Start local gunicorn server",
    )
    @conditional_argument(
        pip_installed_bundle_path is None, "bundle-path", type=click.STRING
    )
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
    def serve_gunicorn(port, workers, timeout, bento=None, with_conda=False):
        track_cli('serve_gunicorn')
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        if with_conda:
            run_with_conda_env(
                pip_installed_bundle_path,
                'bentoml serve_gunicorn {bento} -p {port} -w {workers} '
                '--timeout {timeout}'.format(
                    bento=bento_service_bundle_path,
                    port=port,
                    workers=workers,
                    timeout=timeout,
                ),
            )
            return

        from bentoml.server.gunicorn_server import GunicornBentoServer

        gunicorn_app = GunicornBentoServer(
            bento_service_bundle_path, port, workers, timeout
        )
        gunicorn_app.run()

    # pylint: enable=unused-variable
    return bentoml_cli


def create_bentoml_cli():
    # pylint: disable=unused-variable

    _cli = create_bento_service_cli()

    # Commands created here aren't mean to be used from generated BentoService CLI when
    # installed as PyPI package. The are only used as part of BentoML cli command.

    config_sub_command = get_configuration_sub_command()
    aws_sagemaker_sub_command = get_aws_sagemaker_sub_command()
    aws_lambda_sub_command = get_aws_lambda_sub_command()
    deployment_sub_command = get_deployment_sub_command()
    add_bento_sub_command(_cli)
    _cli.add_command(config_sub_command)
    _cli.add_command(aws_sagemaker_sub_command)
    _cli.add_command(aws_lambda_sub_command)
    _cli.add_command(deployment_sub_command)

    return _cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
