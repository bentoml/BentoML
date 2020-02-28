import subprocess
import logging
import uuid
import os

import docker

from bentoml import BentoService, api
from bentoml.deployment.aws_lambda import _cleanup_s3_bucket_if_exist
from bentoml.deployment.utils import ensure_docker_available_or_raise
from bentoml.handlers import DataframeHandler
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.client import YataiClient

logger = logging.getLogger('bentoml.test')

GRPC_PORT = '50051'
GRPC_CHANNEL_ADDRESS = f'127.0.0.1:{GRPC_PORT}'


def wait_until_container_is_running(container_name):
    docker_client = docker.from_env()
    container_is_not_running = True
    while container_is_not_running:
        logger.debug('Fetching running container list')
        container_list = docker_client.containers.list(filters={'status': 'running'})
        for container in container_list:
            if container.name == container_name:
                container_is_not_running = False
    return


def create_test_postgres():
    ensure_docker_available_or_raise()
    temp_dir = TempDirectory()
    temp_dir.create()
    container_name = 'e2e-yatai-pg-docker'

    command = [
        'docker',
        'run',
        '--rm',
        '--name',
        container_name,
        '-e',
        'POSTGRES_PASSWORD=postgres',
        '-p',
        '5432:5432',
        '-v',
        f'{temp_dir.path}:/var/lib/postgresql/data',
        'postgres',
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_until_container_is_running(container_name)
    return (
        docker_proc,
        temp_dir,
        'postgresql://postgres:postgres@localhost:5432/bentoml',
    )


def create_test_s3_bucket():
    random_hash = uuid.uuid4().hex[:6]
    bucket_name = f'e2e-yatai-server-{random_hash}'
    create_s3_bucket_if_not_exists(bucket_name, 'us-west-2')
    return bucket_name, f's3://{bucket_name}/repo'


def delete_test_postgres(docker_proc, temp_dir):
    docker_proc.terminate()
    temp_dir.cleanup()


def delete_test_s3_bucket(name):
    _cleanup_s3_bucket_if_exist(name, 'us-west-2')


class BentoServiceForYataiTest(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return 'cat'


def delete_bento_service(bento_tag):
    command = ['bentoml', 'delete', bento_tag, '-y']
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_generate_temporary_env_with_channel_address(),
    )
    stdout = proc.stdout.read().decode('utf-8')
    return stdout


def get_bento_service(bento_tag):
    os.environ['BENTOML__YATAI_SERVICE__CHANNEL_ADDRESS'] = GRPC_CHANNEL_ADDRESS
    yatai_client = YataiClient()
    bento_name, bento_version = bento_tag.split(':')
    get_result = yatai_client.repository.get(bento_name, bento_version)
    del os.environ['BENTOML__YATAI_SERVICE__CHANNEL_ADDRESS']
    return get_result


def _generate_temporary_env_with_channel_address(channel_address=None):
    env_copy = os.environ.copy()
    env_copy['BENTOML__YATAI_SERVICE__CHANNEL_ADDRESS'] = GRPC_CHANNEL_ADDRESS
    return env_copy


def run_bento_service_prediction(bento_tag, data):
    command = ['bentoml', 'run', bento_tag, 'predict', '--input', data]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_generate_temporary_env_with_channel_address(),
    )
    stdout = proc.stdout.read().decode('utf-8')
    return stdout


def save_bento_service_with_channel_address():
    os.environ['BENTOML__YATAI_SERVICE__CHANNEL_ADDRESS'] = GRPC_CHANNEL_ADDRESS
    svc = BentoServiceForYataiTest()
    svc.save()
    del os.environ['BENTOML__YATAI_SERVICE__CHANNEL_ADDRESS']
    return svc.name, svc.version
