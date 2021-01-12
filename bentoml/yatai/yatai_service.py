import atexit
import logging
import os
import subprocess
import time
from concurrent import futures

import certifi
import click

from bentoml import config
from bentoml.configuration import get_debug_mode
from bentoml.exceptions import BentoMLException
from bentoml.yatai.utils import ensure_node_available_or_raise, parse_grpc_url


def get_yatai_service(
    channel_address=None,
    access_token=None,
    db_url=None,
    repo_base_url=None,
    s3_endpoint_url=None,
    default_namespace=None,
):
    channel_address = channel_address or config('yatai_service').get('url')
    access_token = access_token or config('yatai_service').get('access_token')
    channel_address = channel_address.strip()
    if channel_address:
        # Lazily import grpcio for YataiSerivce gRPC related actions
        import grpc
        from bentoml.yatai.proto.yatai_service_pb2_grpc import YataiStub
        from bentoml.yatai.client.interceptor import header_client_interceptor

        if any([db_url, repo_base_url, s3_endpoint_url, default_namespace]):
            logger.warning(
                "Using remote YataiService at `%s`, local YataiService configs "
                "including db_url, repo_base_url, s3_endpoint_url and default_namespace"
                "will all be ignored.",
                channel_address,
            )

        logger.debug("Connecting YataiService gRPC server at: %s", channel_address)
        scheme, addr = parse_grpc_url(channel_address)
        header_adder_interceptor = header_client_interceptor.header_adder_interceptor(
            'access_token', access_token
        )
        if scheme in ('grpcs', 'https'):
            tls_root_ca_cert = (
                config().get('yatai_service', 'tls_root_ca_cert')
                # Adding also prev. name to ensure that old configurations do not break.
                or config().get('yatai_service', 'client_certificate_file')
                or certifi.where()  # default: Mozilla ca cert
            )
            tls_client_key = config().get('yatai_service', 'tls_client_key') or None
            tls_client_cert = config().get('yatai_service', 'tls_client_cert') or None
            with open(tls_root_ca_cert, 'rb') as fb:
                ca_cert = fb.read()
            if tls_client_key:
                with open(tls_client_key, 'rb') as fb:
                    tls_client_key = fb.read()
            if tls_client_cert:
                with open(tls_client_cert, 'rb') as fb:
                    tls_client_cert = fb.read()
            credentials = grpc.ssl_channel_credentials(
                ca_cert, tls_client_key, tls_client_cert
            )
            channel = grpc.intercept_channel(
                grpc.secure_channel(addr, credentials), header_adder_interceptor
            )
        else:
            channel = grpc.intercept_channel(
                grpc.insecure_channel(addr), header_adder_interceptor
            )
        return YataiStub(channel)
    else:
        from bentoml.yatai.yatai_service_impl import get_yatai_service_impl

        LocalYataiService = get_yatai_service_impl()

        logger.debug("Creating local YataiService instance")
        return LocalYataiService(
            db_url=db_url,
            repo_base_url=repo_base_url,
            s3_endpoint_url=s3_endpoint_url,
            default_namespace=default_namespace,
        )


def start_yatai_service_grpc_server(
    db_url, repo_base_url, grpc_port, ui_port, with_ui, s3_endpoint_url, base_url
):
    # Lazily import grpcio for YataiSerivce gRPC related actions
    import grpc
    from bentoml.yatai.yatai_service_impl import get_yatai_service_impl
    from bentoml.yatai.proto.yatai_service_pb2_grpc import add_YataiServicer_to_server
    from bentoml.yatai.proto.yatai_service_pb2_grpc import YataiServicer

    YataiServicerImpl = get_yatai_service_impl(YataiServicer)
    yatai_service = YataiServicerImpl(
        db_url=db_url, repo_base_url=repo_base_url, s3_endpoint_url=s3_endpoint_url,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_YataiServicer_to_server(yatai_service, server)
    debug_mode = get_debug_mode()
    if debug_mode:
        try:
            logger.debug('Enabling gRPC server reflection for debugging')
            from grpc_reflection.v1alpha import reflection
            from bentoml.yatai.proto import yatai_service_pb2

            SERVICE_NAMES = (
                yatai_service_pb2.DESCRIPTOR.services_by_name['Yatai'].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(SERVICE_NAMES, server)
        except ImportError:
            logger.debug(
                'Failed to enable gRPC server reflection, missing required package: '
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
        yatai_grpc_server_address = f'localhost:{grpc_port}'
        async_start_yatai_service_web_ui(
            yatai_grpc_server_address, ui_port, web_ui_log_path, debug_mode, base_url
        )

    # We don't import _echo function from click_utils because of circular dep
    click.echo(
        f'* Starting BentoML YataiService gRPC Server\n'
        f'* Debug mode: { "on" if debug_mode else "off"}\n'
        f'''* Web UI: {f"running on http://127.0.0.1:{ui_port}/{base_url}"
        if (with_ui and base_url!=".")
        else f"running on http://127.0.0.1:{ui_port}" if with_ui else "off"}\n'''
        f'* Running on 127.0.0.1:{grpc_port} (Press CTRL+C to quit)\n'
        f'* Help and instructions: '
        f'https://docs.bentoml.org/en/latest/guides/yatai_service.html\n'
        f'{f"* Web server log can be found here: {web_ui_log_path}" if with_ui else ""}'
        f'\n-----\n'
        f'* Usage in Python:\n'
        f'*  bento_svc.save(yatai_url="127.0.0.1:{grpc_port}")\n'
        f'*  bentoml.yatai.client.get_yatai_client("127.0.0.1:{grpc_port}").repository.'
        f'list()\n'
        f'* Usage in CLI:\n'
        f'*  bentoml list --yatai-url=127.0.0.1:{grpc_port}\n'
        f'*  bentoml containerize IrisClassifier:latest --yatai-url=127.0.0.1:'
        f'{grpc_port}\n'
        f'*  bentoml push IrisClassifier:20200918001645_CD2886 --yatai-url=127.0.0.1:'
        f'{grpc_port}\n'
        f'*  bentoml pull IrisClassifier:20200918001645_CD2886 --yatai-url=127.0.0.1:'
        f'{grpc_port}\n'
        f'*  bentoml retrieve IrisClassifier:20200918001645_CD2886 '
        f'--yatai-url=127.0.0.1:{grpc_port} --target_dir="/tmp/foo/bar"\n'
        f'*  bentoml delete IrisClassifier:20200918001645_CD2886 '
        f'--yatai-url=127.0.0.1:{grpc_port}\n'
        # TODO: simplify the example usage here once related documentation is ready
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
    yatai_server_address, ui_port, base_log_path, debug_mode, web_prefix_path
):
    if ui_port is not None:
        ui_port = ui_port if isinstance(ui_port, str) else str(ui_port)
    web_ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web'))
    web_prefix_path = web_prefix_path.strip("/")
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
                web_prefix_path,
            ]
        else:
            web_ui_command = [
                'node',
                'dist/bundle.js',
                yatai_server_address,
                ui_port,
                base_log_path,
                web_prefix_path,
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
            web_prefix_path,
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


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
logger = logging.getLogger(__name__)
