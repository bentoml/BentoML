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

import atexit
import logging
import os
import subprocess
import time
from concurrent import futures

import certifi
import click
import grpc

from bentoml import config
from bentoml.exceptions import BentoMLException
from bentoml.proto.yatai_service_pb2_grpc import add_YataiServicer_to_server
from bentoml.utils.usage_stats import track_server
from bentoml.yatai.utils import ensure_node_available_or_raise, parse_grpc_url

logger = logging.getLogger(__name__)
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def get_yatai_service(
    channel_address=None,
    db_url=None,
    repo_base_url=None,
    s3_endpoint_url=None,
    default_namespace=None,
):
    channel_address = channel_address or config('yatai_service').get('url')
    channel_address = channel_address.strip()
    if channel_address:
        from bentoml.proto.yatai_service_pb2_grpc import YataiStub

        if any([db_url, repo_base_url, s3_endpoint_url, default_namespace]):
            logger.warning(
                "Using remote YataiService at `%s`, local YataiService configs "
                "including db_url, repo_base_url, s3_endpoint_url and default_namespace"
                "will all be ignored.",
                channel_address,
            )

        logger.debug("Connecting YataiService gRPC server at: %s", channel_address)
        scheme, addr = parse_grpc_url(channel_address)

        if scheme in ('grpcs', 'https'):
            client_cacert_path = (
                config().get('yatai_service', 'client_certificate_file')
                or certifi.where()  # default: Mozilla ca cert
            )
            with open(client_cacert_path, 'rb') as ca_cert_file:
                ca_cert = ca_cert_file.read()
            credentials = grpc.ssl_channel_credentials(ca_cert, None, None)
            channel = grpc.secure_channel(addr, credentials)
        else:
            channel = grpc.insecure_channel(addr)
        return YataiStub(channel)
    else:
        from bentoml.yatai.yatai_service_impl import YataiService

        logger.debug("Creating local YataiService instance")
        return YataiService(
            db_url=db_url,
            repo_base_url=repo_base_url,
            s3_endpoint_url=s3_endpoint_url,
            default_namespace=default_namespace,
        )


def start_yatai_service_grpc_server(
    db_url, repo_base_url, grpc_port, ui_port, with_ui, s3_endpoint_url
):
    track_server('yatai-service-grpc-server')
    from bentoml.yatai.yatai_service_impl import YataiService

    yatai_service = YataiService(
        db_url=db_url, repo_base_url=repo_base_url, s3_endpoint_url=s3_endpoint_url,
    )
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
        web_ui_log_path = os.path.join(
            config("logging").get("BASE_LOG_DIR"),
            config('logging').get("yatai_web_server_log_filename"),
        )

        ensure_node_available_or_raise()
        yatai_grpc_server_addess = f'localhost:{grpc_port}'
        async_start_yatai_service_web_ui(
            yatai_grpc_server_addess, ui_port, web_ui_log_path, debug_mode
        )

    # We don't import _echo function from click_utils because of circular dep
    click.echo(
        f'* Starting BentoML YataiService gRPC Server\n'
        f'* Debug mode: { "on" if debug_mode else "off"}\n'
        f'* Web UI: {f"running on http://127.0.0.1:{ui_port}" if with_ui else "off"}\n'
        f'* Running on 127.0.0.1:{grpc_port} (Press CTRL+C to quit)\n'
        f'* Usage:\n'
        f'*  Set config: `bentoml config set yatai_service.url=127.0.0.1:{grpc_port}`\n'
        f'*  Set env var: `export BENTOML__YATAI_SERVICE__URL=127.0.0.1:{grpc_port}`\n'
        f'* Help and instructions: '
        f'https://docs.bentoml.org/en/latest/guides/yatai_service.html\n'
        f'{f"* Web server log can be found here: {web_ui_log_path}" if with_ui else ""}'
    )

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info("Terminating YataiService gRPC server..")
        server.stop(grace=None)


def _is_web_server_debug_tools_available(root_dir):
    return (
        os.path.exists(os.path.join(root_dir, 'node_modules/.bin', 'concurrently'))
        and os.path.exists(os.path.join(root_dir, 'node_modules/.bin', 'ts-node'))
        and os.path.exists(os.path.join(root_dir, 'node_modules/.bin', 'nodemon'))
    )


def async_start_yatai_service_web_ui(
    yatai_server_address, ui_port, base_log_path, debug_mode
):
    if ui_port is not None:
        ui_port = ui_port if isinstance(ui_port, str) else str(ui_port)
    web_ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web'))
    if debug_mode:
        # Only when src/index.ts exists, we will run dev (nodemon)
        if os.path.exists(
            os.path.join(web_ui_dir, 'src/index.ts')
        ) and _is_web_server_debug_tools_available(web_ui_dir):
            web_ui_command = [
                'npm',
                'run',
                'dev',
                '--',
                yatai_server_address,
                ui_port,
                base_log_path,
            ]
        else:
            web_ui_command = [
                'node',
                'dist/bundle.js',
                yatai_server_address,
                ui_port,
                base_log_path,
            ]
    else:
        if not os.path.exists(os.path.join(web_ui_dir, 'dist', 'bundle.js')):
            raise BentoMLException(
                'Yatai web client built is missing. '
                'Please run `npm run build` in the bentoml/yatai/web directory '
                'and then try again'
            )
        web_ui_command = [
            'node',
            'dist/bundle.js',
            yatai_server_address,
            ui_port,
            base_log_path,
        ]

    web_proc = subprocess.Popen(
        web_ui_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=web_ui_dir
    )

    is_web_proc_running = web_proc.poll() is None
    if not is_web_proc_running:
        web_proc_output = web_proc.stdout.read().decode('utf-8')
        logger.error(f'return code: {web_proc.returncode} {web_proc_output}')
        raise BentoMLException('Yatai web ui did not start properly')

    atexit.register(web_proc.terminate)


__all__ = [
    "get_yatai_service",
    "start_yatai_service_grpc_server",
    "async_start_yatai_service_web_ui",
]
