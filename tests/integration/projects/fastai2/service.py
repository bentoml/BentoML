import pathlib
import sys

import numpy as np

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.fastai import FastaiModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([FastaiModelArtifact('model')])
class FastaiClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_data = df.to_numpy().astype(np.float32)
        _, _, output = self.artifacts.model.predict(input_data)

        return output.squeeze().item()


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    bento_dist_path = sys.argv[2]
    service = FastaiClassifier()

    from model.model import Loss, Model  # noqa # pylint: disable=unused-import

    service.artifacts.load_all(artifacts_path)

    pathlib.Path(bento_dist_path).mkdir(parents=True, exist_ok=True)
    service.save_to_dir(bento_dist_path)
