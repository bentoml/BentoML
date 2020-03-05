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
import time
from concurrent import futures

import click
import grpc

from bentoml import config
from bentoml.cli.click_utils import _echo
from bentoml.proto.yatai_service_pb2_grpc import add_YataiServicer_to_server
from bentoml.utils.usage_stats import track_cli, track_server
from bentoml.yatai import get_yatai_service

logger = logging.getLogger(__name__)


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def start_yatai_service_grpc_server(yatai_service, port):
    track_server('yatai-service-grpc-server')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_YataiServicer_to_server(yatai_service, server)
    debug_mode = config().getboolean('core', 'debug')
    if debug_mode:
        try:
            logger.debug('Enabling gRPC server reflection for debugging')
            from grpc_reflection.v1alpha import reflection
            from bentoml.proto import yatai_service_pb2

            SERVICE_NAMES = (
                yatai_service_pb2.DESCRIPTOR.services_by_name['Yatai'].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(SERVICE_NAMES, server)
        except ImportError:
            logger.debug(
                'Failed enabling gRPC server reflection, missing required package: '
                '"pip install grpcio-reflection"'
            )

    server.add_insecure_port(f'[::]:{port}')
    server.start()
    _echo(
        f'* Starting BentoML YataiService gRPC Server\n'
        f'* Debug mode: { "on" if debug_mode else "off"}\n'
        f'* Running on 127.0.0.1:{port} (Press CTRL+C to quit)\n'
        f'* Usage: `bentoml config set yatai_service.url=127.0.0.1:{port}`\n'
        f'* Help and instructions: '
        f'https://docs.bentoml.org/en/latest/concepts/yatai_service.html'
    )
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        _echo("Terminating YataiService gRPC server..")
        server.stop(grace=None)


def add_yatai_service_sub_command(cli):
    # pylint: disable=unused-variable

    @cli.command(help='Start BentoML YataiService for model management and deployment')
    @click.option(
        '--db-url',
        type=click.STRING,
        help='Database URL following RFC-1738, and usually can include username, '
        'password, hostname, database name as well as optional keyword arguments '
        'for additional configuration',
    )
    @click.option(
        '--repo-base-url',
        type=click.STRING,
        help='Base URL for storing saved BentoService bundle files, this can be a '
        'filesystem path(POSIX/Windows), or a S3 URL, usually starts with "s3://"',
    )
    @click.option(
        '--port', type=click.INT, default=50051, help='Port for Yatai server to use'
    )
    def yatai_service_start(db_url, repo_base_url, port):
        track_cli('yatai-service-start')
        yatai_service = get_yatai_service(db_url=db_url, repo_base_url=repo_base_url)
        start_yatai_service_grpc_server(yatai_service, port)
