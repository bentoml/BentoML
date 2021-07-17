import torch  # pylint: disable=import-error

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.pytorch import PytorchLightningModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchLightningModelArtifact("model")])
class PytorchLightningService(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_tensor = torch.from_numpy(df.to_numpy())
        return self.artifacts.model(input_tensor).numpy()
