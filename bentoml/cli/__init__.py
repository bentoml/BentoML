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

from ruamel.yaml import YAML

from bentoml.archive import (
    load,
    load_service_api,
    load_bentoml_config,
    load_bento_service_metadata,
)
from bentoml.server import BentoAPIServer, get_docs
from bentoml.cli.click_utils import BentoMLCommandGroup, conditional_argument, _echo
from bentoml.cli.deployment import get_deployment_sub_command
from bentoml.cli.config import get_configuration_sub_command
from bentoml.utils import Path, ProtoMessageToDict
from bentoml.utils.log import configure_logging
from bentoml.utils.usage_stats import track_cli


def escape_shell_params(param):
    k, v = param.split('=')
    v = re.sub(r'([^a-zA-Z0-9])', r'\\\1', v)
    return '{}={}'.format(k, v)


def create_bento_service_cli(archive_path=None):
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
            configure_logging(logging.DEBUG)
        elif quiet:
            configure_logging(logging.ERROR)
        else:
            configure_logging()  # use default setting in local bentoml.cfg

    # Example Usage: bentoml API_NAME /SAVED_ARCHIVE_PATH --input=INPUT
    @bentoml_cli.command(
        default_command=True,
        default_command_usage="API_NAME BENTO_ARCHIVE_PATH --input=INPUT",
        default_command_display_name="<API_NAME>",
        short_help="Run API function",
        help="Run a API defined in saved BentoArchive with cli args as input",
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @click.argument("api-name", type=click.STRING)
    @conditional_argument(archive_path is None, "archive-path", type=click.STRING)
    @click.option(
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    @click.pass_context
    def run(ctx, api_name, archive_path=archive_path, with_conda=False):
        if with_conda:
            config = load_bentoml_config(archive_path)
            metadata = config['metadata']
            env_name = metadata['service_name'] + '_' + metadata['service_version']

            yaml = YAML()
            yaml.default_flow_style = False
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            env_path = tmpf.name
            yaml.dump(config['env']['conda_env'], Path(env_path))

            pip_req = os.path.join(archive_path, 'requirements.txt')

            subprocess.call(
                'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
                'parameter requires conda but it\'s not installed."; exit 1; }} && '
                'conda env update -n {env_name} -f {env_file} && '
                'conda init bash && '
                'eval "$(conda shell.bash hook)" && '
                'conda activate {env_name} && '
                '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
                'dependencies."; }} &&'
                'bentoml {api_name} {archive_path} {args}'.format(
                    env_name=env_name,
                    env_file=env_path,
                    archive_path=archive_path,
                    api_name=api_name,
                    args=' '.join(map(escape_shell_params, ctx.args)),
                    pip_req=pip_req,
                ),
                shell=True,
            )
            return

        track_cli('run')

        api = load_service_api(archive_path, api_name)
        api.handle_cli(ctx.args)

    # Example Usage: bentoml info /SAVED_ARCHIVE_PATH
    @bentoml_cli.command(
        help="List all APIs defined in the BentoService loaded from archive.",
        short_help="List APIs",
    )
    @conditional_argument(archive_path is None, "archive-path", type=click.STRING)
    def info(archive_path=archive_path):
        """
        List all APIs defined in the BentoService loaded from archive
        """
        track_cli('info')
        bento_service_metadata_pb = load_bento_service_metadata(archive_path)
        output = json.dumps(ProtoMessageToDict(bento_service_metadata_pb), indent=2)
        _echo(output)

    # Example usage: bentoml open-api-spec /SAVED_ARCHIVE_PATH
    @bentoml_cli.command(
        name="open-api-spec",
        help="Display API specification JSON in Open-API format",
        short_help="Display OpenAPI/Swagger JSON specs",
    )
    @conditional_argument(archive_path is None, "archive-path", type=click.STRING)
    def open_api_spec(archive_path=archive_path):
        track_cli('open-api-spec')
        bento_service = load(archive_path)

        _echo(json.dumps(get_docs(bento_service), indent=2))

    # Example Usage: bentoml serve ./SAVED_ARCHIVE_PATH --port=PORT
    @bentoml_cli.command(
        help="Start REST API server hosting BentoService loaded from archive",
        short_help="Start local rest server",
    )
    @conditional_argument(archive_path is None, "archive-path", type=click.STRING)
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
    def serve(port, archive_path=archive_path, with_conda=False):
        if with_conda:
            config = load_bentoml_config(archive_path)
            metadata = config['metadata']
            env_name = metadata['service_name'] + '_' + metadata['service_version']
            pip_req = os.path.join(archive_path, 'requirements.txt')

            subprocess.call(
                'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
                'parameter requires conda but it\'s not installed."; exit 1; }} && '
                'conda env update -n {env_name} -f {env_file} && '
                'conda init bash && '
                'eval "$(conda shell.bash hook)" && '
                'conda activate {env_name} && '
                '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
                'dependencies."; }} &&'
                'bentoml serve {archive_path} --port {port}'.format(
                    env_name=env_name,
                    env_file=os.path.join(archive_path, 'environment.yml'),
                    archive_path=archive_path,
                    port=port,
                    pip_req=pip_req,
                ),
                shell=True,
            )
            return

        track_cli('serve')

        bento_service = load(archive_path)
        server = BentoAPIServer(bento_service, port=port)
        server.start()

    # Example Usage:
    # bentoml serve-gunicorn ./SAVED_ARCHIVE_PATH --port=PORT --workers=WORKERS
    @bentoml_cli.command(
        help="Start REST API gunicorn server hosting BentoService loaded from archive",
        short_help="Start local gunicorn server",
    )
    @conditional_argument(archive_path is None, "archive-path", type=click.STRING)
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
        port, workers, timeout, archive_path=archive_path, with_conda=False
    ):
        if with_conda:
            config = load_bentoml_config(archive_path)
            metadata = config['metadata']
            env_name = metadata['service_name'] + '_' + metadata['service_version']
            pip_req = os.path.join(archive_path, 'requirements.txt')

            subprocess.call(
                'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
                'parameter requires conda but it\'s not installed."; exit 1; }} && '
                'conda env update -n {env_name} -f {env_file} && '
                'conda init bash && '
                'eval "$(conda shell.bash hook)" && '
                'conda activate {env_name} && '
                '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
                'dependencies."; }} &&'
                'bentoml serve_gunicorn {archive_path} -p {port} -w {workers} '
                '--timeout {timeout}'.format(
                    env_name=env_name,
                    env_file=os.path.join(archive_path, 'environment.yml'),
                    archive_path=archive_path,
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

        gunicorn_app = GunicornBentoServer(archive_path, port, workers, timeout)
        gunicorn_app.run()

    # pylint: enable=unused-variable
    return bentoml_cli


def create_bentoml_cli():
    _cli = create_bento_service_cli()

    # Commands created here aren't mean to be used from generated service archive. They
    # are used as part of BentoML cli commands only.

    deployment_sub_command = get_deployment_sub_command()
    config_sub_command = get_configuration_sub_command()
    _cli.add_command(config_sub_command)
    _cli.add_command(deployment_sub_command)

    return _cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
