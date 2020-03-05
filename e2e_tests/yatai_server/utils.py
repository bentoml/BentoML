import subprocess
import contextlib
import logging
import os
import time

from bentoml import BentoService, api
from bentoml.handlers import DataframeHandler
from bentoml.yatai.client import YataiClient

logger = logging.getLogger('bentoml.test')

GRPC_PORT = '50051'
GRPC_CHANNEL_ADDRESS = f'127.0.0.1:{GRPC_PORT}'


class BentoServiceForYataiTest(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return 'cat'


def get_bento_service(bento_name, bento_version):
    yatai_client = YataiClient()
    get_result = yatai_client.repository.get(bento_name, bento_version)
    return get_result


def run_bento_service_prediction(bento_tag, data):
    command = ['bentoml', 'run', bento_tag, 'predict', '--input', data]
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ,
    )
    stdout = proc.stdout.read().decode('utf-8')
    return stdout


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


@contextlib.contextmanager
def start_yatai_server(db_url=None, repo_base_url=None, port=50051):
    yatai_server_command = ['bentoml', 'yatai-service-start']
    if db_url:
        yatai_server_command.extend(['--db-url', db_url])
    if repo_base_url:
        yatai_server_command.extend(['--repo-base-url', repo_base_url])
    try:
        proc = subprocess.Popen(
            yatai_server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        yield f"localhost:{port}"
    finally:
        logger.info('Shutting down YataiServer')
        proc.terminate()
        logger.info('Printing YataiServer log:')
        server_std_out = proc.stdout.read().decode('utf-8')
        logger.info(server_std_out)
