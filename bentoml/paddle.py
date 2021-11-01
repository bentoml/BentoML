import os
import re
import typing as t
from distutils.dir_util import copy_tree

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from ._internal.utils import LazyLoader
from .exceptions import MissingDependencyException

_paddle_exc = """\
Instruction for installing `paddlepaddle`:
- CPU support only: `pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple`
- GPU support: (latest version): `python -m pip install paddlepaddle-gpu==2.1.3.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html`.
    For other version of CUDA or different platforms refers to https://www.paddlepaddle.org.cn/ for more information
"""

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import paddle
    import paddle.inference
    import paddle.nn
    import paddlehub as hub
    from _internal.models.store import ModelStore, StoreCtx
    from paddle.fluid.dygraph.dygraph_to_static.program_translator import StaticFunction
    from paddle.static import InputSpec

try:
    import paddle
    import paddle.inference
    import paddle.nn
except ImportError:  # pragma: no cover
    raise MissingDependencyException(_paddle_exc)

_hub_exc = (
    """\
`paddlehub` is required to use `bentoml.paddle.import_from_paddlehub()`. Make sure to have `paddlepaddle`
 installed beforehand. Install `paddlehub` with `pip install paddlehub`.
"""
    + _paddle_exc
)

hub = LazyLoader("hub", globals(), "paddlehub", exc_msg=_hub_exc)

PADDLE_MODEL_EXTENSION: str = ".pdmodel"
PADDLE_PARAMS_EXTENSION: str = ".pdiparams"


def _clean_name(name: str) -> str:  # pragma: no cover
    return re.sub(r"\W|^(?=\d)-", "_", name)


@inject
def load(
    tag: str,
    config: t.Optional["paddle.inference.Config"] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "paddle.inference.Predictor":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        config (`paddle.inference.Config`, `optional`, default to `None`):
            Model config to be used to create a `Predictor` instance.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `paddle.inference.Predictor` from BentoML modelstore.

    Examples::
    """

    # https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/analysis_config.cc
    info = model_store.get(tag)
    if config is None:
        config = paddle.inference.Config(
            os.path.join(info.path, f"{SAVE_NAMESPACE}{PADDLE_MODEL_EXTENSION}"),
            os.path.join(info.path, f"{SAVE_NAMESPACE}{PADDLE_PARAMS_EXTENSION}"),
        )
        config.enable_memory_optim()
    return paddle.inference.create_predictor(config)


# class PaddleHubModel(Model):
#     def __init__(self, model: PathType, metadata: t.Optional[GenericDictType] = None):
#         if os.path.isdir(model):
#             module = hub.Module(directory=model)
#             self._dir = str(model)
#         else:
#             # TODO: refactor to skip init Module in memory
#             module = hub.Module(name=model)
#             self._dir = ""
#         super(PaddleHubModel, self).__init__(module, metadata=metadata)
#
#     def save(self, path: PathType) -> None:
#         if self._dir != "":
#             copy_tree(self._dir, str(path))
#         else:
#             self._model.save_inference_model(path)
#
#     @classmethod
#     def load(cls, path: PathType) -> t.Any:
#         # https://github.com/PaddlePaddle/PaddleHub/blob/release/v2.1/paddlehub/module/module.py#L233
#         # we don't have a custom name, so this should be stable
#         # TODO: fix a bug when loading as module
#         model_fpath = os.path.join(path, "__model__")
#         if os.path.isfile(model_fpath):
#             import paddlehub.module.manager as manager  # noqa
#
#             man = manager.LocalModuleManager()
#             module_class = man.install(directory=str(path))
#             module_class.directory = str(path)
#             return module_class
#         else:
#             # custom module that installed from directory
#             return hub.Module(directory=str(path))


def _internal_save(
    name: str,
    model: t.Union[str, "paddle.nn.Layer", "paddle.inference.Predictor"],
    *,
    input_spec: t.Optional[t.Union[t.List[InputSpec], t.Tuple[InputSpec, ...]]],
    metadata: t.Optional[t.Dict[str, t.Any]],
    model_store: "ModelStore",
) -> str:
    context = {"paddlepaddle": paddle.__version__}
    if isinstance(model, str):
        context["paddlehub"] = hub.__version__
    with model_store.register(
        name,
        module=__name__,
        framework_context=context,
        metadata=metadata,
    ) as ctx:  # type: StoreCtx
        if isinstance(model, str):
            ...
        else:
            paddle.jit.save(
                model, os.path.join(ctx.path, SAVE_NAMESPACE), input_spec=input_spec
            )
        _tag = ctx.tag  # type: str
        return _tag


@inject
def save(
    name: str,
    model: t.Union["paddle.nn.Layer", "paddle.inference.Predictor", "StaticFunction"],
    *,
    input_spec: t.Optional[t.Union[t.List[InputSpec], t.Tuple[InputSpec, ...]]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`Union["paddle.nn.Layer", "paddle.inference.Predictor", "StaticFunction"]`):
            Instance of `paddle.nn.Layer`, decorated functions, or `paddle.inference.Predictor` to be saved.
        input_spec (`Union[List[InputSpec], Tuple[InputSpec, ...]]`, `optional`, default to `None`):
            Describes the input of the saved model's forward method, which can be described by InputSpec
             or example Tensor. Moreover, we support to specify non-tensor type argument, such as `int`,
             `float`, `string`, or `list`/`dict` of them. If `None`, all input variables of the original
             Layer's forward method would be the inputs of the saved model. Generally this is NOT RECOMMENDED
             to use unless you know what you are doing.
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
    """
    return _internal_save(
        name=name,
        model=model,
        input_spec=input_spec,
        metadata=metadata,
        model_store=model_store,
    )


@inject
def import_from_paddlehub(
    model_name: str,
    name: t.Optional[str] = None,
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Import models from PaddleHub and save it under BentoML modelstore.

    Args:
        model_name (`str`):
            Name for a PaddleHub model. This can be either path to a Hub module, model name,
             for both v2 and v1.
        name (`str`, `optional`, default to `None`):
            Name for given model instance. This should pass Python identifier check. If not
             specified, then BentoML will save under `model_name`.
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::

    """
    return _internal_save(
        name=_clean_name(model_name) if not name else name,
        model=model_name,
        input_spec=None,
        metadata=metadata,
        model_store=model_store,
    )


class _PaddlePaddleRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_store.get(self.name).tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        ...

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data) -> t.Any:  # type: ignore[override]
        ...


@inject
def load_runner(
    tag: str,
    *,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PaddlePaddleRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.paddle.load_runner` implements a Runner class that
    wrap around a PaddlePaddle Predictor, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.paddle` model

    Examples::
    """
    return _PaddlePaddleRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
