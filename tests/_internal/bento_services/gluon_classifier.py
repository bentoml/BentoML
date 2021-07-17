import mxnet as mx  # pylint: disable=import-error

from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput
from bentoml.gluon import GluonModelArtifact


@env(infer_pip_packages=True)
@artifacts([GluonModelArtifact("model")])
class GluonClassifier(BentoService):
    @api(input=JsonInput(), batch=False)
    def predict(self, request):
        nd_input = mx.nd.array(request)
        return self.artifacts.model(nd_input).asnumpy()
