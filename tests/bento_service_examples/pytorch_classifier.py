import numpy
import torch  # pylint: disable=import-error

import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.adapters import DataframeInput


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([PytorchModelArtifact('model')])
class PytorchClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        input_data = df.to_numpy().astype(numpy.float32)
        input_tensor = torch.from_numpy(input_data)
        output = self.artifacts.model(input_tensor)

        return output.unsqueeze(dim=0).item()
