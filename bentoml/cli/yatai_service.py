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
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai import get_yatai_service

logger = logging.getLogger(__name__)


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def start_yatai_grpc_server(yatai_service, port):
    track_cli('yatai-server-start')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_YataiServicer_to_server(yatai_service, server)
    if config().getboolean('core', 'debug'):
        logger.debug('Add reflection API for debugging gRPC server')

        from grpc_reflection.v1alpha import reflection
        from bentoml.proto import yatai_service_pb2

        SERVICE_NAMES = (
            yatai_service_pb2.DESCRIPTOR.services_by_name['Yatai'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f'[::]:{port}')
    server.start()
    _echo(f'Yatai Server is running on 127.0.0.1:{port}  (Press CTRL+C to quit)')
    _echo(
        f'Run `bentoml config set yatai_service.channel_address=127.0.0.1:{port}` '
        f'in another terminal before use any BentoML tooling.'
    )
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        track_cli('yatai-server-stop')
        _echo("Terminating Yatai server...")
        server.stop(grace=None)


def add_yatai_service_sub_command(cli):
    # pylint: disable=unused-variable

    @cli.command(help='Start local Yatai server')
    @click.option(
        '--db-url', type=click.STRING, help='Database URL for storing BentoML metadata'
    )
    @click.option(
        '--repo-base-url',
        type=click.STRING,
        help='Remote URL address for storing BentoServices, such as S3',
    )
    @click.option(
        '--port', type=click.INT, default=50051, help='Port for Yatai server to use'
    )
    def yatai_service_start(db_url, repo_base_url, port):
        yatai_service = get_yatai_service(db_url=db_url, repo_base_url=repo_base_url)
        start_yatai_grpc_server(yatai_service, port)
