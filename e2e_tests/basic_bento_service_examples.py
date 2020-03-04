from bentoml import BentoService, api
from bentoml.handlers import JsonHandler


class BasicBentoService(BentoService):
    @api(JsonHandler)
    def predict(self, data):
        return 'cat'


class UpdatedBasicBentoService(BentoService):
    @api(JsonHandler)
    def predict(self, data):
        return 'dog'
