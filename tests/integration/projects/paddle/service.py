import pathlib
import sys

import numpy as np

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.paddle import PaddlePaddleModel


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PaddlePaddleModel("model")])
class PaddleService(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_data = df.to_numpy().astype(np.float32)

        predictor = self.artifacts.model
        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[0])

        input_handle.reshape(input_data.shape)
        input_handle.copy_from_cpu(input_data)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        return output_data


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    bento_dist_path = sys.argv[2]

    service = PaddleService()
    service.artifacts.load_all(artifacts_path)

    pathlib.Path(bento_dist_path).mkdir(parents=True, exist_ok=True)
    service.save_to_dir(bento_dist_path)
