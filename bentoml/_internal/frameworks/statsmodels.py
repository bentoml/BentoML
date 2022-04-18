import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

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

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..models import ModelStore

    ModelType = t.Any


MODULE_NAME = "bentoml.statsmodels"

_exc_msg = """\
`pandas` is required by `bentoml.statsmodels`, install pandas with
 `pip install pandas`. For more information, refer to
 https://pandas.pydata.org/docs/getting_started/install.html
"""


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "ModelType":
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

    """
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")
    return sm.load(model_file)


def save(
    name: str,
    model: "ModelType",
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Union[None, t.Dict[str, t.Union[str, int]]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Any`):
            Instance of model to be saved.
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

    with bentoml.models.create(
        name,
        module=__name__,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as _model:

        model.save(_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"))

        return _model.tag


class _StatsModelsRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        name: t.Optional[str] = None,
    ):
        super().__init__(tag, name=name)
        self._predict_fn_name = predict_fn_name

    @property
    def _num_threads(self) -> int:
        # NOTE: Statsmodels currently doesn't use GPU, so return max. no. of CPU's.
        return max(round(self.resource_quota.cpu), 1)

    @property
    def num_replica(self) -> int:
        # NOTE: Statsmodels currently doesn't use GPU, so just return 1.
        return 1

    def _setup(self) -> None:
        self._model = load(self._tag, model_store=self.model_store)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    def _run_batch(self, input_data: t.Union["ext.NpNDArray", "ext.PdDataFrame"]) -> t.Any:  # type: ignore[override] # noqa
        # TODO: type hint return type.
        parallel, p_func, _ = parallel_func(  # type: ignore[arg-type]
            self._predict_fn,
            n_jobs=self._num_threads,
            verbose=0,
        )
        return parallel(p_func(i) for i in input_data)[0]  # type: ignore


def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "predict",
    name: t.Optional[str] = None,
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

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.statsmodels` model

    Examples:

    .. code-block:: python

        import bentoml
        import pandas as pd

        runner = bentoml.statsmodels.load_runner("holtswinter")
        runner.run_batch(pd.DataFrame("/path/to/data"))

    """
    return _StatsModelsRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        name=name,
    )
