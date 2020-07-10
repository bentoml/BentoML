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
        help='Base URL for storing BentoML saved bundle files, this can be a filesystem'
        'path(POSIX/Windows), or a S3 URL, usually starting with "s3://"',
        envvar='BENTOML_REPO_BASE_URL',
    )
    @click.option(
        '--grpc-port',
        type=click.INT,
        default=50051,
        help='Port to run YataiService gRPC server',
        envvar='BENTOML_GRPC_PORT',
    )
    @click.option(
        '--ui-port',
        type=click.INT,
        default=3000,
        help='Port to run YataiService Web UI server',
        envvar='BENTOML_WEB_UI_PORT',
    )
    @click.option(
        '--ui/--no-ui',
        default=True,
        help='Run YataiService with or without Web UI, when running with --no-ui, it '
        'will only run the gRPC server',
        envvar='BENTOML_ENABLE_WEB_UI',
    )
    @click.option(
        '--s3-endpoint-url',
        type=click.STRING,
        help='S3 Endpoint URL is used for deploying with storage services that are '
        'compatible with Amazon S3, such as MinIO',
        envvar='BENTOML_S3_ENDPOINT_URL',
    )
    def yatai_service_start(
        db_url, repo_base_url, grpc_port, ui_port, ui, s3_endpoint_url
    ):
        from bentoml.yatai.yatai_service import start_yatai_service_grpc_server

        start_yatai_service_grpc_server(
            db_url, repo_base_url, grpc_port, ui_port, ui, s3_endpoint_url
        )
