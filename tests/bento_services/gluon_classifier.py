from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.gluon import GluonModelArtifact
import mxnet as mx  # pylint: disable=import-error


@env(infer_pip_packages=True)
@artifacts([GluonModelArtifact("model")])
class GluonClassifier(BentoService):
    @api(input=JsonInput(), batch=False)
    def predict(self, request):
        nd_input = mx.nd.array(request)
        return self.artifacts.model(nd_input).asnumpy()
