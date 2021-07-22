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
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import paddle
    import paddle.inference as pi
except ImportError:
    raise MissingDependencyException(
        "paddlepaddle is required by PaddlePaddleModel and PaddleHubModel"
    )


class PaddlePaddleModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`onnx` models.

    Args:
        model (`Union[paddle.nn.Layer, paddle.inference.Predictor]`):
            Every PaddlePaddle model is of type :obj:`paddle.nn.Layer`
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`paddle` is required by PaddlePaddleModel and PaddleHubModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    PADDLE_MODEL_EXTENSION: str = ".pdmodel"
    PADDLE_PARAMS_EXTENSION: str = ".pdiparams"

    _model: t.Union[paddle.nn.Layer, paddle.inference.Predictor]

    def __init__(
        self,
        model: t.Union[paddle.nn.Layer, paddle.inference.Predictor],
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PaddlePaddleModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> paddle.inference.Predictor:
        # https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/analysis_config.cc
        config = pi.Config(
            cls.get_path(path, cls.PADDLE_MODEL_EXTENSION),
            cls.get_path(path, cls.PADDLE_PARAMS_EXTENSION),
        )
        config.enable_memory_optim()
        return pi.create_predictor(config)

    def save(self, path: PathType) -> None:
        # Override the model path if temp dir was set
        # TODO(aarnphm): What happens if model is a paddle.inference.Predictor?
        paddle.jit.save(self._model, self.get_path(path))


class PaddleHubModel(ModelArtifact):
    pass
