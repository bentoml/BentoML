import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..utils import LazyLoader
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import numpy as np
    from pandas.core.frame import DataFrame

    from ..models import ModelStore
else:
    np = LazyLoader("np", globals(), "numpy")

try:
    import catboost as cbt
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """catboost is required in order to use module `bentoml.catboost`, install
        catboost with `pip install catboost`. For more information, refers to
        https://catboost.ai/docs/concepts/python-installation.html
        """
    )

MODULE_NAME = "bentoml.catboost"

# TODO: support cbt.Pool runner io container

CATBOOST_EXT = "cbm"


def _get_model_info(
    tag: Tag,
    model_params: t.Optional[t.Dict[str, t.Union[str, int]]],
    model_store: "ModelStore",
) -> t.Tuple["Model", str, t.Dict[str, t.Any]]:
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )

    model_file = model.path_of(f"{SAVE_NAMESPACE}.{CATBOOST_EXT}")
    _model_params: t.Dict[str, t.Union[str, int]] = (
        dict() if not model_params else model_params
    )
    for key, value in model.info.options.items():
        if key not in _model_params:
            _model_params[key] = value  # pragma: no cover

    return model, model_file, _model_params


def _load_helper(
    model_file: str, model_params: t.Optional[t.Dict[str, t.Union[str, int]]]
) -> t.Union[
    cbt.core.CatBoost,
    cbt.core.CatBoostClassifier,
    cbt.core.CatBoostRegressor,
]:

    if model_params is not None:
        model_type = model_params["model_type"]

        if model_type == "classifier":
            model = cbt.core.CatBoostClassifier()
        elif model_type == "regressor":
            model = cbt.core.CatBoostRegressor()
        else:
            model = cbt.core.CatBoost()
    else:
        model = cbt.core.CatBoost()

    _m: t.Union[
        cbt.core.CatBoost,
        cbt.core.CatBoostClassifier,
        cbt.core.CatBoostRegressor,
    ] = model.load_model(model_file)
    return _m


@inject
def load(
    tag: t.Union[str, Tag],
    model_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Union[
    cbt.core.CatBoost, cbt.core.CatBoostClassifier, cbt.core.CatBoostRegressor
]:
    """
    Load a CatBoost model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_params (:code:`Dict[str, Union[str, Any]]`, `optional`, default to :code:`None`): Parameters for
            a CatBoost model. Following parameters can be specified:
                - model_type(:code:`str`): :obj:`classifier` (`CatBoostClassifier`) or :obj:`regressor` (`CatBoostRegressor`)
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`Union[catboost.core.CatBoost, catboost.core.CatBoostClassifier, catboost.core.CatBoostRegressor]`: one of :code:`catboost.core.CatBoostClassifier`,
        :code:`catboost.core.CatBoostRegressor` or :code:`catboost.core.CatBoost` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        booster = bentoml.catboost.load("my_model:latest", model_params=dict(model_type="classifier"))
    """  # noqa

    _, _model_file, _model_params = _get_model_info(tag, model_params, model_store)

    return _load_helper(_model_file, _model_params)


@inject
def save(
    name: str,
    model: t.Union[
        cbt.core.CatBoost,
        cbt.core.CatBoostClassifier,
        cbt.core.CatBoostRegressor,
    ],
    *,
    model_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]] = None,
    model_export_parameters: t.Optional[t.Dict[str, t.Any]] = None,
    model_pool: t.Optional["cbt.core.Pool"] = None,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[catboost.core.CatBoost, catboost.core.CatBoostClassifier, catboost.CatBoostRegressor]`):
            Instance of model to be saved
        model_params (:code:`Dict[str, Union[str, Any]]`, `optional`, default to :code:`None`):
            Parameters for a CatBoost model. Following parameters can be specified:
                - model_type(:code:`str`): :obj:`classifier` (`CatBoostClassifier`) or :obj:`regressor` (`CatBoostRegressor`)
        model_export_parameters (:code:`Dict[str, Union[str, Any]]`, `optional`, default to :code:`None`):
            Export parameters for given model.
        model_pool (:code:`cbt.core.Pool`, `optional`, default to :code:`None`):
            CatBoost data pool for given model.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        from sklearn.datasets import load_breast_cancer

        import catboost as cbt

        # read in data
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target

        # create and train model
        model = cbt.CatBoostClassifier(iterations=2,
                                   depth=2,
                                   learning_rate=1,
                                   loss_function='Logloss',
                                   verbose=True)

        model.fit(X, y)
        ...

        tag = bentoml.catboost.save("my_catboost_model", model, model_params=dict(model_type="classifier"))

        # load the booster back:
        loaded = bentoml.catboost.load("my_catboost_model:latest")
        # or:
        loaded = bentoml.catboost.load(tag)

    """  # noqa
    if not model_params:
        model_params = {}

    if "model_type" not in model_params:
        model_params["model_type"] = "classifier"

    context = {
        "framework_name": "catboost",
        "pip_dependencies": [f"catboost=={get_pkg_version('catboost')}"],
    }
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=model_params,
        metadata=metadata,
        context=context,
    )

    path = _model.path_of(f"{SAVE_NAMESPACE}.{CATBOOST_EXT}")
    format_ = CATBOOST_EXT
    model.save_model(
        path,
        format=format_,
        export_parameters=model_export_parameters,
        pool=model_pool,
    )

    _model.save(model_store)
    return _model.tag


class _CatBoostRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        predict_fn_name: str,
        model_params: t.Optional[t.Dict[str, t.Union[str, int]]],
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        model_info, model_file, _model_params = _get_model_info(
            tag, model_params, model_store
        )
        super().__init__(name, resource_quota, batch_options)
        self._model_info = model_info
        self._model_file = model_file
        self._predict_fn_name = predict_fn_name
        self._model_params = _model_params

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    @property
    def num_replica(self) -> int:
        return int(round(self.resource_quota.cpu))

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        self._model = _load_helper(self._model_file, self._model_params)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", "DataFrame", cbt.Pool],
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        res = self._predict_fn(inputs)
        return np.asarray(res)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "predict",
    *,
    model_params: t.Union[None, t.Dict[str, t.Union[str, int]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    name: t.Optional[str] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
) -> "_CatBoostRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.catboost.load_runner` implements a Runner class that
    wrap around a CatBoost model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`predict`):
            Options for inference functions. `predict` are the default function.
        model_params (:code:`Dict[str, Union[str, Any]]`, `optional`, default to :code:`None`): Parameters for
            a CatBoost model. Following parameters can be specified:
                - model_type(:code:`str`): :obj:`classifier` (`CatBoostClassifier`) or :obj:`regressor` (`CatBoostRegressor`)
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.catboost` model

    Examples:

    .. code-block:: python

        import catboost as cbt
        import pandas as pd

        input_data = pd.read_csv("/path/to/csv")
        runner = bentoml.catboost.load_runner("my_model:latest"")
        runner.run(cbt.Pool(input_data))
    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _CatBoostRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        model_params=model_params,
        model_store=model_store,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
    )
