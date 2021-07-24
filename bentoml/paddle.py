import os
import typing as t
from distutils.dir_util import copy_tree

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import paddle
    import paddle.inference as pi
except ImportError:
    raise MissingDependencyException(
        "paddlepaddle is required by PaddlePaddleModel and PaddleHubModel"
    )

try:
    import paddlehub as hub
except ImportError:
    hub = None


class PaddlePaddleModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`paddlepaddle` models.

    Args:
        model (`Union[paddle.nn.Layer, paddle.inference.Predictor]`):
            Every PaddlePaddle model is of type :obj:`paddle.nn.Layer`
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`paddlepaddle` is required by PaddlePaddleModel

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
    """
    Model class for saving/loading :obj:`paddlehub` models.

    Args:
        model (`Union[str, os.PathLike]`):
            Either a custom :obj:`paddlehub.Module` directory, or
            pretrained model from PaddleHub registry.
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`paddlehub` and :obj:`paddlepaddle` are required by PaddleHubModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(self, model: PathType, metadata: t.Optional[MetadataType] = None):
        if hub is None:
            raise MissingDependencyException("paddlehub is required by PaddleHubModel")
        if os.path.isdir(model):
            module = hub.Module(directory=model)
            self._dir = str(model)
        else:
            module = hub.Module(name=model)
            self._dir = ""
        super(PaddleHubModel, self).__init__(module, metadata=metadata)

    def save(self, path: PathType) -> None:
        if self._dir != "":
            copy_tree(self._dir, str(path))
        else:
            self._model.save_inference_model(str(path))

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        if hub is None:
            raise MissingDependencyException("paddlehub is required by PaddleHubModel")
        return hub.Module(directory=str(path))
