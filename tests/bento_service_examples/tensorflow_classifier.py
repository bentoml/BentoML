import bentoml
from bentoml.adapters import TfTensorInput
from bentoml.artifact import TensorflowSavedModelArtifact


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class TensorflowClassifier(bentoml.BentoService):
    @bentoml.api(input=TfTensorInput())
    def predict(self, tensor):
        return self.artifacts.model(tensor)
