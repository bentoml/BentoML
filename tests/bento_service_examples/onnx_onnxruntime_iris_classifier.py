import numpy

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.onnx import OnnxModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([OnnxModelArtifact('model', backend='onnxruntime')])
class OnnxIrisClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_data = df.to_numpy().astype(numpy.float32)
        input_name = self.artifacts.model.get_inputs()[0].name
        output_name = self.artifacts.model.get_outputs()[0].name
        return self.artifacts.model.run([output_name], {input_name: input_data})[0]
