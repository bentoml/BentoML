import os
import typing as t
from distutils.dir_util import copy_tree

import bentoml._internal.constants as const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

_paddle_exc = const.IMPORT_ERROR_MSG.format(
    fwr="paddlepaddle",
    module=__name__,
    inst="`pip install paddlepaddle` for CPU options"
    " or `pip install paddlepaddle-gpu` for GPU options.",
)

_hub_exc = const.IMPORT_ERROR_MSG.format(
    fwr="paddlehub",
    module=__name__,
    inst="`pip install paddlepaddle`," " then `pip install paddlehub`",
)


if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import paddle
    import paddle.inference as pi
    import paddlehub as hub
else:
    paddle = LazyLoader("paddle", globals(), "paddle", exc_msg=_paddle_exc)
    pi = LazyLoader("pi", globals(), "paddle.inference", exc_msg=_paddle_exc)
    hub = LazyLoader("hub", globals(), "paddlehub", exc_msg=_hub_exc)


class PaddlePaddleModel(Model):
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

    One then can define :code:`bento.py`::

        TODO:
    """

    PADDLE_MODEL_EXTENSION: str = ".pdmodel"
    PADDLE_PARAMS_EXTENSION: str = ".pdiparams"

    _model: t.Union["paddle.nn.Layer", "pi.Predictor"]

    def __init__(
        self,
        model: t.Union["paddle.nn.Layer", "pi.Predictor"],
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PaddlePaddleModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(  # pylint: disable=arguments-differ
        cls, path: PathType, config: t.Optional["pi.Config"] = None
    ) -> "pi.Predictor":
        # https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/analysis_config.cc
        if config is None:
            config = pi.Config(
                os.path.join(path, f"{MODEL_NAMESPACE}{cls.PADDLE_MODEL_EXTENSION}"),
                os.path.join(path, f"{MODEL_NAMESPACE}{cls.PADDLE_PARAMS_EXTENSION}"),
            )
            config.enable_memory_optim()
        return pi.create_predictor(config)

    def save(self, path: PathType) -> None:
        # Override the model path if temp dir was set
        # TODO(aarnphm): What happens if model is a paddle.inference.Predictor?
        paddle.jit.save(self._model, os.path.join(path, MODEL_NAMESPACE))


class PaddleHubModel(Model):
    """
    Model class for saving/loading :obj:`paddlehub` models.

    Args:
        model (`Union[str, bytes, os.PathLike]`):
            Either a custom :obj:`paddlehub.Module` directory, or
            pretrained model from PaddleHub registry.
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`paddlehub` and :obj:`paddlepaddle` are required by PaddleHubModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:

    """

    def __init__(self, model: PathType, metadata: t.Optional[MetadataType] = None):
        if os.path.isdir(model):
            module = hub.Module(directory=model)
            self._dir = str(model)
        else:
            # TODO: refactor to skip init Module in memory
            module = hub.Module(name=model)
            self._dir = ""
        super(PaddleHubModel, self).__init__(module, metadata=metadata)

    def save(self, path: PathType) -> None:
        if self._dir != "":
            copy_tree(self._dir, str(path))
        else:
            self._model.save_inference_model(path)

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        # https://github.com/PaddlePaddle/PaddleHub/blob/release/v2.1/paddlehub/module/module.py#L233
        # we don't have a custom name, so this should be stable
        # TODO: fix a bug when loading as module
        model_fpath = os.path.join(path, "__model__")
        if os.path.isfile(model_fpath):
            import paddlehub.module.manager as manager  # noqa

            man = manager.LocalModuleManager()
            module_class = man.install(directory=str(path))
            module_class.directory = str(path)
            return module_class
        else:
            # custom module that installed from directory
            return hub.Module(directory=str(path))
