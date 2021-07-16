import numpy

import bentoml
import torch  # pylint: disable=import-error
from bentoml.adapters import DataframeInput
from bentoml.pytorch import PytorchModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact("model")])
class PytorchClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_data = df.to_numpy().astype(numpy.float32)
        input_tensor = torch.from_numpy(input_data)
        output = self.artifacts.model(input_tensor)

        return output.unsqueeze(dim=0).item()
