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


import logging

import click

from bentoml.cli.click_utils import CLI_COLOR_ERROR, _echo
from bentoml.exceptions import BentoMLException
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai import start_yatai_service_grpc_server

logger = logging.getLogger(__name__)


def add_yatai_service_sub_command(cli):
    # pylint: disable=unused-variable

    @cli.command(help='Start BentoML YataiService for model management and deployment')
    @click.option(
        '--db-url',
        type=click.STRING,
        help='Database URL following RFC-1738, and usually can include username, '
        'password, hostname, database name as well as optional keyword arguments '
        'for additional configuration',
        envvar='BENTOML_DB_URL',
    )
    @click.option(
        '--repo-base-url',
        type=click.STRING,
        help='Base URL for storing saved BentoService bundle files, this can be a '
        'filesystem path(POSIX/Windows), or a S3 URL, usually starts with "s3://"',
        envvar='BENTOML_REPO_BASE_URL',
    )
    @click.option(
        '--grpc-port', type=click.INT, default=50051, help='Port for Yatai server'
    )
    @click.option(
        '--ui-port', type=click.INT, default=3000, help='Port for Yatai web UI'
    )
    @click.option(
        '--ui/--no-ui',
        default=True,
        help='Start BentoML YataiService without Web UI',
        envvar='BENTOML_ENABLE_WEB_UI',
    )
    def yatai_service_start(db_url, repo_base_url, grpc_port, ui_port, ui):
        track_cli('yatai-service-start')
        try:
            start_yatai_service_grpc_server(
                db_url, repo_base_url, grpc_port, ui_port, ui
            )
        except BentoMLException as e:
            _echo(f'Yatai gRPC server failed: {str(e)}', CLI_COLOR_ERROR)
