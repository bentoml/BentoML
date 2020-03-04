import subprocess
import logging
import os

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
