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


import logging
import os
import subprocess
import time
from concurrent import futures

import grpc

from bentoml import config
from bentoml.proto.yatai_service_pb2_grpc import add_YataiServicer_to_server
from bentoml.utils.usage_stats import track_server


logger = logging.getLogger(__name__)
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def get_yatai_service(
    channel_address=None, db_url=None, repo_base_url=None, default_namespace=None
):
    channel_address = channel_address or config().get('yatai_service', 'url')
    if channel_address:
        import grpc
        from bentoml.proto.yatai_service_pb2_grpc import YataiStub

        if db_url is not None:
            logger.warning(
                "Config 'db_url' is ignored in favor of remote YataiService at `%s`",
                channel_address,
            )
        if repo_base_url is not None:
            logger.warning(
                "Config 'repo_base_url:%s' is ignored in favor of remote YataiService "
                "at `%s`",
                repo_base_url,
                channel_address,
            )
        if default_namespace is not None:
            logger.warning(
                "Config 'default_namespace:%s' is ignored in favor of remote "
                "YataiService at `%s`",
                default_namespace,
                channel_address,
            )
        logger.debug("Using BentoML with remote Yatai server: %s", channel_address)

        channel = grpc.insecure_channel(channel_address)
        return YataiStub(channel)
    else:
        from bentoml.yatai.yatai_service_impl import YataiService

        logger.debug("Using BentoML with local Yatai server")

        default_namespace = default_namespace or config().get(
            'deployment', 'default_namespace'
        )
        repo_base_url = repo_base_url or config().get('default_repository_base_url')
        db_url = db_url or config().get('db', 'url')

        return YataiService(
            db_url=db_url,
            repo_base_url=repo_base_url,
            default_namespace=default_namespace,
        )


def start_yatai_service_grpc_server(db_url, repo_base_url, grpc_port, ui_port, with_ui):
    track_server('yatai-service-grpc-server')
    yatai_service = get_yatai_service(db_url=db_url, repo_base_url=repo_base_url)
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
    server.add_insecure_port(f'[::]:{grpc_port}')
    server.start()
    if with_ui:
        yatai_grpc_server_addess = f'localhost:{grpc_port}'
        proc = async_start_yatai_service_web_ui(
            yatai_grpc_server_addess, ui_port, debug_mode
        )
    logger.info(
        f'* Starting BentoML YataiService gRPC Server\n'
        f'* Debug mode: { "on" if debug_mode else "off"}\n'
        f'* Web UI: {f"running on http://127.0.0.1:{ui_port}" if with_ui else "off"}\n'
        f'* Running on 127.0.0.1:{grpc_port} (Press CTRL+C to quit)\n'
        f'* Usage: `bentoml config set yatai_service.url=127.0.0.1:{grpc_port}`\n'
        f'* Help and instructions: '
        f'https://docs.bentoml.org/en/latest/concepts/yatai_service.html'
    )
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info("Terminating YataiService gRPC server..")
        if with_ui and proc:
            proc.terminate()
        server.stop(grace=None)


def async_start_yatai_service_web_ui(yatai_server_address, ui_port, debug_mode):
    if ui_port is not None:
        ui_port = ui_port if isinstance(ui_port, str) else str(ui_port)
    web_ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web'))
    if debug_mode:
        # WIP need to find way to include port and yatai server address into watch
        web_ui_command = ['npm', 'watch']
    else:
        # NOTE, we need to make sure we build dist before we start this
        if not os.path.exists(os.path.join(web_ui_dir, 'dist')):
            build_web_dist = subprocess.Popen(
                ['npm', 'build'],
                cwd=web_ui_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.debug(build_web_dist.stdout.read().decode('utf-8'))
        web_ui_command = ['node', 'dist/index.js', yatai_server_address, ui_port]
    web_proc = subprocess.Popen(
        web_ui_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=web_ui_dir,
    )
    return web_proc


__all__ = [
    "get_yatai_service",
    "start_yatai_service_grpc_server",
    "async_start_yatai_service_web_ui",
]
