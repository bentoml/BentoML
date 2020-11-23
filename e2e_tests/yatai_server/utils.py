import subprocess
import contextlib
import logging
import os
import uuid

import psutil

from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.client import YataiClient

logger = logging.getLogger('bentoml.test')

GRPC_PORT = '50051'
GRPC_CHANNEL_ADDRESS = f'127.0.0.1:{GRPC_PORT}'


def get_bento_service_info(bento_name, bento_version):
    yatai_client = YataiClient()
    get_result = yatai_client.repository.get(f'{bento_name}:{bento_version}')
    return get_result


def execute_bentoml_run_command(bento_tag, data, api="predict"):
    command = ['bentoml', 'run', bento_tag, api, '--input', data, "-q"]
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ,
    )
    stdout = proc.stdout.read().decode('utf-8')
    return stdout


def execute_bentoml_retrieve_command(bento_tag):
    dir_name = uuid.uuid4().hex[:8]
    with TempDirectory() as temp_dir:
        command = [
            'bentoml',
            'retrieve',
            bento_tag,
            '--target_dir',
            f'{temp_dir}/{dir_name}',
        ]
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ
        )
        stdout = proc.stdout.read().decode('utf-8')
        print(stdout)
        print(proc.stderr.read().decode('utf-8'))
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
        [env.pop(k, None) for k in remove]  # pylint: disable=expression-not-assigned
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]  # pylint: disable=expression-not-assigned


def kill_process(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


@contextlib.contextmanager
def local_yatai_server(db_url=None, repo_base_url=None, port=50051):
    yatai_server_command = ['bentoml', 'yatai-service-start']
    if db_url:
        yatai_server_command.extend(['--db-url', db_url])
    if repo_base_url:
        yatai_server_command.extend(['--repo-base-url', repo_base_url])
    try:
        proc = subprocess.Popen(
            yatai_server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        yatai_service_url = f"localhost:{port}"
        logger.info(f'Setting config yatai_service.url to: {yatai_service_url}')
        with modified_environ(BENTOML__YATAI_SERVICE__URL=yatai_service_url):
            yield yatai_service_url
    finally:
        logger.info('Shutting down YataiServer gRPC server and node web server')
        kill_process(proc.pid)
