import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner

from ..models import JSON_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..runner.utils import Params
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from ..models import ModelStore

try:
    import mxnet
    import mxnet.gluon as gluon
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """mxnet is required in order to use module `bentoml.gluon`, install
        mxnet with `pip install mxnet`. For more information, refers
        to https://mxnet.apache.org/versions/master/get_started?  """
    )


MODULE_NAME = "bentoml.gluon"


@inject
def load(
    tag: t.Union[str, Tag],
    mxnet_ctx: t.Optional[mxnet.context.Context] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> gluon.Block:
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        mxnet_ctx (`mxnet.context.Context`, `optional`, default to :code:``cpu``):
            Device type to cast model. Default behaviour similar
            to :obj:`torch.device("cuda")` Options: "cuda" or "cpu".
            If `None` is specified then return default `config.MODEL.DEVICE`
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`mxnet.gluon.Block`: an instance of :obj:`mxnet.gluon.Block` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        model = bentoml.gluon.load(tag)

    """  # noqa

    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
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
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`mxnet.gluon.HybridBlock`):
            Instance of gluon.HybridBlock model to be saved.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import mxnet
        import mxnet.gluon as gluon
        import bentoml


        def train_gluon_classifier() -> gluon.nn.HybridSequential:
            net = mxnet.gluon.nn.HybridSequential()
            net.hybridize()
            net.forward(mxnet.nd.array(0))
            return net

        model = train_gluon_classifier()

        tag = bentoml.gluon.save("gluon_block", model)
    """  # noqa

    context: t.Dict[str, t.Any] = {
        "framework_name": "gluon",
        "pip_dependencies": [f"mxnet=={get_pkg_version('mxnet')}"],
    }
    options: t.Dict[str, t.Any] = dict()
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=options,
        context=context,
        metadata=metadata,
    )

    model.export(_model.path_of(SAVE_NAMESPACE))

    _model.save(model_store)

    return _model.tag


class _GluonRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        predict_fn_name: str,
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)
        self._tag = tag
        self._predict_fn_name = predict_fn_name
        self._model_store = model_store
        self._ctx = None

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

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
    name: t.Optional[str] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _GluonRunner:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.detectron.load_runner` implements a Runner class that
    wrap around a torch.nn.Module model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`__call__`):
            Options for inference functions. Default to `__call__`
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.detectron` model

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.gluon.load_runner("gluon_block")
        runner.run_batch(data)

    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _GluonRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
