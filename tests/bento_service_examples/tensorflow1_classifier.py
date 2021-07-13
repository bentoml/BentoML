import sys

import bentoml
from bentoml.adapters import TfTensorInput
from bentoml.tensorflow import TensorflowSavedModelArtifact

if "tensorflow" not in sys.modules:
    import tensorflow

    tensorflow.enable_eager_execution()


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class Tensorflow1Classifier(bentoml.BentoService):
    @bentoml.api(input=TfTensorInput(), batch=True)
    def predict(self, tensor):
        import tensorflow as tf

        tf.enable_eager_execution()

        pred_func = self.artifacts.model.signatures['serving_default']
        return pred_func(tensor)['prediction']
