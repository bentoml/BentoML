import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import Provide, inject

from bentoml._internal.runner.utils import Params

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import JSON_EXT, SAVE_NAMESPACE, Model
from ._internal.runner import Runner
from ._internal.types import Tag
from .exceptions import BentoMLException, MissingDependencyException

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from ._internal.models import ModelStore

try:
    import mxnet
    import mxnet.gluon as gluon
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """mxnet is required in order to use module `bentoml.gluon`, install
        mxnet with `pip install mxnet`. For more information, refers
        to https://mxnet.apache.org/versions/master/get_started?  """
    )


@inject
def load(
    tag: t.Union[str, Tag],
    mxnet_ctx: t.Optional[mxnet.context.Context] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> gluon.Block:
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        mxnet_ctx (mxnet.context.Context, `optional`, default to ``cpu``):
            Device type to cast model. Default behaviour similar
             to :obj:`torch.device("cuda")` Options: "cuda" or "cpu".
             If None is specified then return default config.MODEL.DEVICE
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `torch.nn.Module`

    Examples::
        TODO
    """  # noqa

    model = model_store.get(tag)
    if model.info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with"
            f" module {model.info.module},"
            f" failed loading with {__name__}."
        )

    json_path: str = model.path_of(f"{SAVE_NAMESPACE}-symbol{JSON_EXT}")
    params_path: str = model.path_of(f"{SAVE_NAMESPACE}-0000.params")

    return gluon.SymbolBlock.imports(json_path, ["data"], params_path, ctx=mxnet_ctx)


@inject
def save(
    name: str,
    model: gluon.HybridBlock,
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`mxnet.gluon.HybridBlock`):
            Instance of gluon.HybridBlock model to be saved.
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        TODO

    """  # noqa

    context: t.Dict[str, t.Any] = {"gluon": mxnet.__version__}
    options: t.Dict[str, t.Any] = dict()
    _model = Model.create(
        name,
        module=__name__,
        options=options,
        framework_context=context,
        metadata=metadata,
    )

    model.export(_model.path_of(SAVE_NAMESPACE))

    _model.save(model_store)

    return _model.tag


class _GluonRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(str(tag), resource_quota, batch_options)
        self._tag = Tag.from_taglike(tag)
        self._predict_fn_name = predict_fn_name
        self._model_store = model_store
        self._ctx = None

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        if self.resource_quota.on_gpu:
            ctx = mxnet.gpu()
        else:
            ctx = mxnet.cpu()
        self._ctx = ctx
        self._model = load(self._tag, ctx, self._model_store)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(
        self,
        *args: t.Union[np.ndarray, mxnet.ndarray.NDArray],
        **kwargs: t.Union[np.ndarray, mxnet.ndarray.NDArray],
    ) -> np.ndarray:
        params = Params[t.Union[np.ndarray, mxnet.ndarray.NDArray]](*args, **kwargs)
        if isinstance(params.sample, np.ndarray):
            params = params.map(lambda i: mxnet.nd.array(i, ctx=self._ctx))
        elif isinstance(params.sample, mxnet.ndarray.NDArray):
            params = params.map(lambda i: i.as_in_context(self._ctx))
        else:
            raise TypeError(
                "`_run_batch` of {self.__class__.__name__} only takes "
                "mxnet.nd.array or numpt.ndarray as input parameters"
            )
        res = self._predict_fn(*params.args, **params.kwargs)
        return res.asnumpy()


@inject
def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "__call__",
    *,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _GluonRunner:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.detectron.load_runner` implements a Runner class that
    wrap around a torch.nn.Module model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `__call__`):
            Options for inference functions. Default to `__call__`
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.detectron` model

    Examples:
        TODO
    """  # noqa
    return _GluonRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
