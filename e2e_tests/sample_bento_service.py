import logging

from bentoml import BentoService, api
from bentoml.adapters import JsonInput


logger = logging.getLogger('bentoml.test')


class SampleBentoService(BentoService):
    @api(input=JsonInput())
    def predict(self, data):
        logger.info(f"SampleBentoService predict API received data {data}")
        return 'cat'


class UpdatedSampleBentoService(BentoService):
    @api(input=JsonInput())
    def predict(self, data):
        logger.info(f"UpdatedSampleBentoService predict API received data {data}")
        return 'dog'
