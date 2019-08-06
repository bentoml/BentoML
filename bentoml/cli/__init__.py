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

import json
import click
import logging

from bentoml.archive import load, load_service_api
from bentoml.server import BentoAPIServer, get_docs
from bentoml.server.gunicorn_server import GunicornBentoServer
from bentoml.cli.click_utils import BentoMLCommandGroup, conditional_argument, _echo
from bentoml.cli.deployment import add_deployment_commands
from bentoml.cli.config import get_configuration_sub_command
from bentoml.utils.log import configure_logging
from bentoml.utils.usage_stats import track_cli


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
    @click.pass_context
    def run(ctx, api_name, archive_path=archive_path):
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
        bento_service = load(archive_path)

        service_apis = bento_service.get_service_apis()
        output = json.dumps(
            dict(
                name=bento_service.name,
                version=bento_service.version,
                apis=[api.name for api in service_apis],
            ),
            indent=2,
        )
        _echo(output)

    # Example usage: bentoml docs /SAVED_ARCHIVE_PATH
    @bentoml_cli.command(
        help="Display API documents in Open API format",
        short_help="Display OpenAPI docs",
    )
    @conditional_argument(archive_path is None, "archive-path", type=click.STRING)
    def docs(archive_path=archive_path):
        track_cli('docs')
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
    def serve(port, archive_path=archive_path):
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
    def serve_gunicorn(port, workers, timeout, archive_path=archive_path):
        track_cli('serve_gunicorn')

        gunicorn_app = GunicornBentoServer(archive_path, port, workers, timeout)
        gunicorn_app.run()

    # pylint: enable=unused-variable
    return bentoml_cli


def create_bentoml_cli():
    _cli = create_bento_service_cli()

    # Commands created here aren't mean to be used from generated service archive. They
    # are used as part of BentoML cli commands only.

    add_deployment_commands(_cli)
    config_sub_command = get_configuration_sub_command()
    _cli.add_command(config_sub_command)
    return _cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
