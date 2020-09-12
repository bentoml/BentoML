import numpy
import pandas as pd

from coremltools.models import MLModel  # pylint: disable=import-error

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.coreml import CoreMLModelArtifact


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([CoreMLModelArtifact('model')])
class CoreMLClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput())
    def predict(self, df: pd.DataFrame) -> float:
        model: MLModel = self.artifacts.model
        input_data = df.to_numpy().astype(numpy.float32)
        output = model.predict({"input": input_data})
        return next(iter(output.values())).item()
