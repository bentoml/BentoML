import numpy

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.onnxmlir import OnnxMlirModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([OnnxMlirModelArtifact('model')])
class OnnxMlirClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_data = df.to_numpy().astype(numpy.float32)
        return self.artifacts.model.run(input_data)
