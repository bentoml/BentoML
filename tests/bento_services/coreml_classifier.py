import numpy
import pandas as pd

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.coreml import CoreMLModelArtifact
from coremltools.models import MLModel  # pylint: disable=import-error


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([CoreMLModelArtifact("model")])
class CoreMLClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame) -> float:
        model: MLModel = self.artifacts.model
        input_data = df.to_numpy().astype(numpy.float32)
        output = model.predict({"input": input_data})
        return next(iter(output.values())).item()
