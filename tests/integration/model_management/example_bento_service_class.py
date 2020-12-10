import logging

from bentoml import BentoService, api, artifacts
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput

logger = logging.getLogger('bentoml.test')


@artifacts([PickleArtifact('model')])
class ExampleBentoService(BentoService):
    @api(input=JsonInput(), batch=False)
    def predict(self, data):
        return data
