import bentoml
from bentoml.adapters import JsonInput, TfTensorInput
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts(
    [
        TensorflowSavedModelArtifact('model1'),
        TensorflowSavedModelArtifact('model2'),
        TensorflowSavedModelArtifact('model3'),
    ]
)
class Tensorflow2Classifier(bentoml.BentoService):
    @bentoml.api(input=TfTensorInput(), batch=True)
    def predict1(self, tensor):
        return self.artifacts.model1(tensor)

    @bentoml.api(input=TfTensorInput(), batch=True)
    def predict2(self, tensor):
        return self.artifacts.model2(tensor)

    @bentoml.api(input=JsonInput(), batch=True)
    def predict3(self, jsons):
        import tensorflow as tf

        tensor = tf.ragged.constant(jsons, dtype=tf.float64)
        return self.artifacts.model3(tensor)
