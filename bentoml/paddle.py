# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import os
import tempfile

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import MissingDependencyException

try:
    import paddle
    import paddle.inference as paddle_infer
except ImportError:
    paddle = None


class PaddlePaddleModelArtifact(ModelArtifact):
    """
    Artifact class for saving and loading PaddlePaddle's PaddleInference model

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: paddle package is required for
                                    PaddlePaddleModelArtifact

    Example usage:

    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> from bentoml import env, artifacts, api, BentoService
    >>> from bentoml.adapters import DataframeInput
    >>> from bentoml.frameworks.paddle import PaddlePaddleModelArtifact
    >>>
    >>> @env(infer_pip_packages=True)
    >>> @artifacts([PaddlePaddleModelArtifact('model')])
    >>> class PaddleService(BentoService):
    >>>    @api(input=DataframeInput(), batch=True)
    >>>    def predict(self, df: pd.DataFrame):
    >>>        input_data = df.to_numpy().astype(np.float32)
    >>>        predictor = self.artifacts.model
    >>>
    >>>        input_names = predictor.get_input_names()
    >>>        input_handle = predictor.get_input_handle(input_names[0])
    >>>        input_handle.reshape(input_data.shape)
    >>>        input_handle.copy_from_cpu(input_data)
    >>>
    >>>        predictor.run()
    >>>
    >>>        output_names = predictor.get_output_names()
    >>>        output_handle = predictor.get_output_handle(output_names[0])
    >>>        output_data = output_handle.copy_to_cpu()
    >>>        return output_data
    >>>
    >>> service = PaddleService()
    >>>
    >>> service.pack('model', model_to_save)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._model = None
        self._predictor = None
        self._get_path = None

        if paddle is None:
            raise MissingDependencyException(
                "paddlepaddle package is required to use PaddlePaddleModelArtifact"
            )

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
        self._model = model
        return self

    def load(self, path):
        model = paddle.jit.load(self._file_path(path))
        model = paddle.jit.to_static(model, input_spec=model._input_spec())
        return self.pack(model)

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def save(self, dst):
        self._save(dst)

    def _save(self, dst):
        # Override the model path if temp dir was set
        self._get_path = self._file_path(dst)
        paddle.jit.save(self._model, self._get_path)

    def get(self):
        # Create predictor, if one doesn't exist, when inference is run
        if not self._predictor:
            # If model isn't saved, save model to a temp dir
            # because predictor init requires the path to a saved model
            if self._get_path is None:
                self._get_path = tempfile.TemporaryDirectory().name
                self._save(self._get_path)

            config = paddle_infer.Config(
                self._get_path + ".pdmodel", self._get_path + ".pdiparams"
            )
            config.enable_memory_optim()
            predictor = paddle_infer.create_predictor(config)
            self._predictor = predictor
        return self._predictor


class PaddleHubModel(ModelArtifact):
    pass
