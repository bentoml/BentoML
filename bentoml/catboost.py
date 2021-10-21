import os
import typing as t

import numpy as np
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import BentoMLException, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import pandas as pd

    from ._internal.models.store import ModelInfo, ModelStore, StoreCtx  # noqa

try:
    import catboost as cbt
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """catboost is required in order to use module `bentoml.catboost`, install
        catboost with `pip install catboost`. For more information, refers to
        https://catboost.ai/docs/concepts/python-installation.html
        """
    )


# TODO: support cbt.Pool runner io container

CATBOOST_EXT = "cbm"

CatBoostModelType = t.TypeVar(
    "CatBoostModelType",
    bound=t.Union[
        cbt.core.CatBoost,
        cbt.core.CatBoostClassifier,
        cbt.core.CatBoostRegressor,
    ],
)


def _get_model_info(
    tag: str,
    model_params: t.Optional[t.Dict[str, t.Union[str, int]]],
    model_store: "ModelStore",
) -> t.Tuple["ModelInfo", str, t.Dict[str, t.Any]]:
    model_info = model_store.get(tag)
    if model_info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with module {model_info.module}, failed loading "
            f"with {__name__}."
        )

    model_file = os.path.join(model_info.path, f"{SAVE_NAMESPACE}.{CATBOOST_EXT}")
    _model_params = dict() if not model_params else model_params
    for key, value in model_info.options.items():
        if key not in _model_params:
            _model_params[key] = value  # pragma: no cover

    return model_info, model_file, _model_params


def _load_helper(
    model_file: str, model_params: t.Optional[t.Dict[str, t.Union[str, int]]]
) -> CatBoostModelType:

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

    _m = model.load_model(model_file)  # type: CatBoostModelType
    return _m


@inject
def load(
    tag: str,
    model_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> CatBoostModelType:

    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_params (`t.Dict[str, t.Union[str, Any]]`):
            Parameters for model. Following parameters can be specified:
                - model_type: "classifier" or "regressor" Determine if the model is
                  a `CatBoostClassifier` or `CatBoostRegressor`
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `catboost.core.CatBoostClassifier` or `catboost.core. CatBoostRegressor`
        from BentoML modelstore.

    Examples::
        import bentoml.catboost
        booster = bentoml.catboost.load(
            "my_model:20201012_DE43A2", model_params=dict(model_type="classifier"))
    """  # noqa

    _, _model_file, _model_params = _get_model_info(tag, model_params, model_store)

    return _load_helper(_model_file, _model_params)


@inject
def save(
    name: str,
    model: CatBoostModelType,
    *,
    model_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]] = None,
    model_export_parameters: t.Optional[t.Dict[str, t.Any]] = None,
    model_pool: t.Optional["cbt.core.Pool"] = None,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Union[catboost.core.CatBoost, catboost.core.CatBoostClassifier, catboost.CatBoostRegressor]`):
            Instance of model to be saved
        model_params (`t.Dict[str, t.Union[str, t.Any]]`, `optional`, default to `None`):
            Parameters for model. Following parameters can be specified:
                - model_type: "classifier" or "regressor" Determine if the model is
                  a `CatBoostClassifier` or `CatBoostRegressor`
        model_export_parameters (`t.Dict[str, t.Union[str, t.Any]]`, `optional`, default to `None`):
            Export parameters for given model.
        model_pool (`cbt.core.Pool`, `optional`, default to `None`):
            CatBoost data pool for given model.
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        from sklearn.datasets import load_breast_cancer

        import catboost as cbt
        from bentoml.catboost import CatBoostClassifier

        # read in data
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target

        # create and trainmodel
        model = CatBoostClassifier(iterations=2,
                                   depth=2,
                                   learning_rate=1,
                                   loss_function='Logloss',
                                   verbose=True)

        model.fit(X, y)
        ...

        tag = bentoml.catboost.save("my_catboost_model", model, model_params=dict(model_type="classifier"))
        # example tag: my_catboost_model:20211001_AB90F5_153BC4

        # load the booster back:
        loaded = bentoml.catboost.load("my_catboost_model:latest") # or
        loaded = bentoml.catboost.load(tag)

    """  # noqa
    if not model_params:
        model_params = {}

    if "model_type" not in model_params:
        model_params["model_type"] = "classifier"

    context = {"catboost": cbt.__version__}
    with model_store.register(
        name,
        module=__name__,
        options=model_params,
        metadata=metadata,
        framework_context=context,
    ) as ctx:  # type: StoreCtx
        path = os.path.join(ctx.path, f"{SAVE_NAMESPACE}.{CATBOOST_EXT}")
        format_ = CATBOOST_EXT
        model.save_model(
            path,
            format=format_,
            export_parameters=model_export_parameters,
            pool=model_pool,
        )
        return ctx.tag


class _CatBoostRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        predict_fn_name: str,
        model_params: t.Optional[t.Dict[str, t.Union[str, int]]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        model_info, model_file, _model_params = _get_model_info(
            tag, model_params, model_store
        )

        self._model_info = model_info
        self._model_file = model_file
        self._predict_fn_name = predict_fn_name
        self._model_params = _model_params

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        return int(round(self.resource_quota.cpu))

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = _load_helper(self._model_file, self._model_params)  # type: ignore[var-annotated] # noqa: E501 # pylint: disable
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self, input_data: t.Union[np.ndarray, "pd.DataFrame", cbt.Pool]
    ) -> np.ndarray:
        # Take a batch type
        # Take a containers
        res = self._predict_fn(input_data)
        return np.asarray(res)


@inject
def load_runner(
    tag: str,
    predict_fn_name: str = "predict",
    *,
    model_params: t.Union[None, t.Dict[str, t.Union[str, int]]] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_CatBoostRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.catboost.load_runner` implements a Runner class that
    wrap around a CatBoost model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `predict`):
            Options for inference functions. `predict` are the default function.
        model_params (`t.Dict[str, t.Union[str, int]]`, default to `None`):
            Parameters for model. Following parameters can be specified:
                - model_type: "classifier" or "regressor" Determine if the model is
                  a `CatBoostClassifier` or `CatBoostRegressor`
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.catboost` model

    Examples::
        import catboost as cbt
        import bentoml.catboost
        import pandas as pd

        input_data = pd.from_csv("/path/to/csv")
        runner = bentoml.catboost.load_runner("my_model:latest"")
        runner.run(cbt.Pool(input_data))
    """  # noqa
    return _CatBoostRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        model_params=model_params,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
