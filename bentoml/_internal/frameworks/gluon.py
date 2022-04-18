import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml._internal.types import LazyType

from ..models import JSON_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..runner.utils import Params
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..models import ModelStore

try:
    import mxnet  # type: ignore
    import mxnet.gluon as gluon  # type: ignore
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


def save(
    name: str,
    model: gluon.HybridBlock,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`mxnet.gluon.HybridBlock`):
            Instance of gluon.HybridBlock model to be saved.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

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
    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        options=options,
        context=context,
        metadata=metadata,
    ) as _model:

        model.export(_model.path_of(SAVE_NAMESPACE))

        return _model.tag


class _GluonRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        name: t.Optional[str] = None,
    ):
        super().__init__(tag, name=name)
        self._predict_fn_name = predict_fn_name
        self._ctx = None

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    def _setup(self) -> None:
        if self.resource_quota.on_gpu:
            ctx = mxnet.gpu()
        else:
            ctx = mxnet.cpu()
        self._ctx = ctx
        self._model = load(self._tag, ctx, model_store=self.model_store)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    def _run_batch(
        self,
        *args: t.Union["ext.NpNDArray", mxnet.ndarray.NDArray],
        **kwargs: t.Union["ext.NpNDArray", mxnet.ndarray.NDArray],
    ) -> "ext.NpNDArray":
        params = Params(*args, **kwargs)
        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(params.sample):
            params = params.map(lambda i: mxnet.nd.array(i, ctx=self._ctx))  # type: ignore
        elif LazyType["mxnet.ndarray.NDArray"](mxnet.ndarray.NDArray).isinstance(
            params.sample
        ):
            params = params.map(lambda i: i.as_in_context(self._ctx))  # type: ignore
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
    name: t.Optional[str] = None,
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

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.detectron` model

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.gluon.load_runner("gluon_block")
        runner.run_batch(data)

    """
    return _GluonRunner(
        name=name,
        tag=tag,
        predict_fn_name=predict_fn_name,
    )
