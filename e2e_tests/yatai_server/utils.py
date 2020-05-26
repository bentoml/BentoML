import subprocess
import logging
import os

from bentoml import BentoService, api
from bentoml.handlers import DataframeHandler
from bentoml.yatai.client import YataiClient

logger = logging.getLogger('bentoml.test')


class BentoServiceForYataiTest(BentoService):
    @api(DataframeHandler)
    def predict(self, df):  # pylint: disable=unused-argument
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
