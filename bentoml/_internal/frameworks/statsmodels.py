import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject
from simple_di import Provide

from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..types import Tag
from ..types import PathType
from ..models import Model
from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..runner import Runner
from ..utils.lazy_loader import LazyLoader
from ..configuration.containers import BentoMLContainer

_MT = t.TypeVar("_MT")

if TYPE_CHECKING:
    import pandas as pd
    from joblib.parallel import Parallel

    from ..models import ModelStore


try:
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.tools.parallel import parallel_func
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """statsmodels is required in order to use bentoml.statsmodels, install
         statsmodels with `pip install statsmodels`. For more information, refer to
         https://www.statsmodels.org/stable/install.html
         """
    )

_exc_msg = """\
`pandas` is required by `bentoml.statsmodels`, install pandas with
 `pip install pandas`. For more information, refer to
 https://pandas.pydata.org/docs/getting_started/install.html
"""
pd = LazyLoader("pd", globals(), "pandas", exc_msg=_exc_msg)  # noqa: F811


def _get_model_info(
    tag: t.Union[str, Tag],
    model_store: "ModelStore",
) -> t.Tuple["Model", PathType]:
    model = model_store.get(tag)
    if model.info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with module {model.info.module}, failed loading"
            f"with {__name__}"
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    return model, model_file


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _MT:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of pickled model from BentoML modelstore.

    Examples::
    """  # noqa
    _, model_file = _get_model_info(tag, model_store)
    _load: t.Callable[[PathType], _MT] = sm.load
    return _load(model_file)


@inject
def save(
    name: str,
    model: _MT,
    *,
    metadata: t.Union[None, t.Dict[str, t.Union[str, int]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Any):
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
    """  # noqa
    context: t.Dict[str, t.Any] = {"statsmodels": statsmodels.__version__}
    _model = Model.create(
        name,
        module=__name__,
        metadata=metadata,
        framework_context=context,
    )

    model.save(_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"))

    _model.save(model_store)
    return _model.tag


class _StatsModelsRunner(Runner):
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
        model_info, model_file = _get_model_info(tag, model_store)
        self._predict_fn_name = predict_fn_name
        self._model_info = model_info
        self._model_file = model_file
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        # NOTE: Statsmodels currently doesn't use GPU, so return max. no. of CPU's.
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        # NOTE: Statsmodels currently doesn't use GPU, so just return 1.
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = sm.load(self._model_file)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: t.Union[np.ndarray, "pd.DataFrame"]) -> t.Any:  # type: ignore[override] # noqa
        # TODO: type hint return type.
        parallel: "Parallel"
        p_func: t.Callable[..., t.Any]
        parallel, p_func, _ = parallel_func(
            self._predict_fn, n_jobs=self.num_concurrency_per_replica, verbose=0
        )
        return parallel(p_func(i) for i in input_data)[0]


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "predict",
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_StatsModelsRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.statsmodels.load_runner` implements a Runner class that
    wrap around a statsmodels instance, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `predict`):
            Options for inference functions
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.statsmodels` model

    Examples::
    """  # noqa
    return _StatsModelsRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
