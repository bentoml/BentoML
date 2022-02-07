import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import PathType
from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..utils.lazy_loader import LazyLoader
from ..configuration.containers import BentoMLContainer

_MT = t.TypeVar("_MT")

if TYPE_CHECKING:
    import pandas as pd
    from joblib.parallel import Parallel

    from ..models import ModelStore


try:
    import statsmodels.api as sm
    from statsmodels.tools.parallel import parallel_func
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """statsmodels is required in order to use bentoml.statsmodels, install
         statsmodels with `pip install statsmodels`. For more information, refer to
         https://www.statsmodels.org/stable/install.html
         """
    )

MODULE_NAME = "bentoml.statsmodels"

_exc_msg = """\
`pandas` is required by `bentoml.statsmodels`, install pandas with
 `pip install pandas`. For more information, refer to
 https://pandas.pydata.org/docs/getting_started/install.html
"""
pd = LazyLoader("pd", globals(), "pandas", exc_msg=_exc_msg)  # noqa: F811


def _get_model_info(
    tag: Tag,
    model_store: "ModelStore",
) -> t.Tuple["Model", PathType]:
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    return model, model_file


@inject
def load(
    tag: Tag,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _MT:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`Any`: an instance of :mod:`statsmodels` that is unpickled from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        model = bentoml.statsmodels.load("holtswinter")

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
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Any`):
            Instance of model to be saved.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml
        import pandas as pd

        import statsmodels
        from statsmodels.tsa.holtwinters import HoltWintersResults
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        def holt_model() -> "HoltWintersResults":
            df: pd.DataFrame = pd.read_csv(
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
            )

            # Taking a test-train split of 80 %
            train = df[0 : int(len(df) * 0.8)]
            test = df[int(len(df) * 0.8) :]

            # Pre-processing the  Month  field
            train.Timestamp = pd.to_datetime(train.Month, format="%m-%d")
            train.index = train.Timestamp
            test.Timestamp = pd.to_datetime(test.Month, format="%m-%d")
            test.index = test.Timestamp

            # fitting the model based on  optimal parameters
            return ExponentialSmoothing(
                np.asarray(train["Sales"]),
                seasonal_periods=7,
                trend="add",
                seasonal="add",
            ).fit()

        # `save` a given classifier and retrieve coresponding tag:
        tag = bentoml.statsmodels.save("holtwinters", HoltWintersResults())

    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "statsmodels",
        "pip_dependencies": [f"statsmodels=={get_pkg_version('statsmodels')}"],
    }
    _model = Model.create(
        name,
        module=__name__,
        metadata=metadata,
        context=context,
    )

    model.save(_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"))

    _model.save(model_store)
    return _model.tag


class _StatsModelsRunner(Runner):
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
        model_info, model_file = _get_model_info(tag, model_store)
        self._predict_fn_name = predict_fn_name
        self._model_info = model_info
        self._model_file = model_file
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    @property
    def _num_threads(self) -> int:
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
            self._predict_fn,
            n_jobs=self._num_threads,
            verbose=0,
        )
        return parallel(p_func(i) for i in input_data)[0]


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "predict",
    name: t.Optional[str] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_StatsModelsRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.statsmodels.load_runner` implements a Runner class that
    wrap around a statsmodels instance, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`predict`):
            Options for inference functions
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.statsmodels` model

    Examples:

    .. code-block:: python

        import bentoml
        import pandas as pd

        runner = bentoml.statsmodels.load_runner("holtswinter")
        runner.run_batch(pd.DataFrame("/path/to/data"))

    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _StatsModelsRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
