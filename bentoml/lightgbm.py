import os
import typing as t

import joblib
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import PKL_EXT, SAVE_NAMESPACE, TXT_EXT
from ._internal.models.store import StoreCtx
from ._internal.runner import Runner
from .exceptions import BentoMLException, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import lightgbm as lgb
    import numpy as np
    from _internal.models.store import ModelInfo, ModelStore

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """lightgbm is required in order to use module `bentoml.lightgbm`, install
        lightgbm with `pip install lightgbm`. For more information, refer to
        https://github.com/microsoft/LightGBM/tree/master/python-package
        """
    )

_LightGBMModelType = t.TypeVar(
    "_LightGBMModelType",
    bound=t.Union[
        "lgb.LGBMModel", "lgb.LGBMClassifier", "lgb.LGBMRegressor", "lgb.LGBMRanker"
    ],
)


def _get_model_info(
    tag: str,
    booster_params: t.Optional[t.Dict[str, t.Union[str, int]]],
    model_store: "ModelStore",
) -> t.Tuple["ModelInfo", str, t.Dict[str, t.Any]]:
    model_info = model_store.get(tag)
    if model_info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with"
            f" module {model_info.module},"
            f" failed loading with {__name__}"
        )
    _fname = (
        f"{SAVE_NAMESPACE}{TXT_EXT}"
        if not model_info.options["sklearn_api"]
        else f"{SAVE_NAMESPACE}{PKL_EXT}"
    )
    model_file = os.path.join(model_info.path, _fname)
    _booster_params = dict() if not booster_params else booster_params
    for key, value in model_info.options.items():
        if key not in _booster_params:
            _booster_params[key] = value

    return model_info, model_file, _booster_params


@inject
def load(
    tag: str,
    booster_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Union["lgb.basic.Booster", _LightGBMModelType]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        booster_params (`t.Dict[str, t.Union[str, int]]`):
            Parameters for boosters. Refers to https://lightgbm.readthedocs.io/en/latest/Parameters.html
            for more information.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `LightGBMModelType` or `"lgb.basic.Booster"` from BentoML modelstore.

    Examples:
        import bentoml.lightgbm
        gbm = bentoml.lightgbm.load("my_lightgbm_model:latest")
    """  # noqa
    _, _model_file, _booster_params = _get_model_info(tag, booster_params, model_store)

    if os.path.splitext(_model_file)[1] == PKL_EXT:
        return joblib.load(_model_file)
    else:
        return lgb.Booster(params=_booster_params, model_file=_model_file)


@inject
def save(
    name: str,
    model: t.Union["lgb.basic.Booster", _LightGBMModelType],
    *,
    booster_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Union["lgb.basic.Booster", LightGBMModelType]`):
            Instance of model to be saved
        booster_params (`t.Dict[str, t.Union[str, int]]`):
            Parameters for boosters. Refers to https://lightgbm.readthedocs.io/en/latest/Parameters.html
            for more information.
        metadata (`t.Union[None, t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples:
        import lightgbm as lgb
        import bentoml.lightgbm
        import pandas as pd

        # load a dataset
        df_train = pd.read_csv("regression.train", header=None, sep="\t")
        df_test = pd.read_csv("regression.test", header=None, sep="\t")

        y_train = df_train[0]
        y_test = df_test[0]
        X_train = df_train.drop(0, axis=1)
        X_test = df_test.drop(0, axis=1)

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": {"l2", "l1"},
            "num_leaves": 31,
            "learning_rate": 0.05,
        }

        # train
        gbm = lgb.train(
            params, lgb_train, num_boost_round=20, valid_sets=lgb_eval
        )

        tag = bentoml.lightgbm.save("my_lightgbm_model", gbm, booster_params=params)
        # example tag: my_lightgbm_model:20211021_80F7DB

        # load the booster back:
        gbm = bentoml.lightgbm.load("my_lightgbm_model:latest")
    """  # noqa
    context = {"lightgbm": lgb.__version__}
    with model_store.register(
        name,
        module=__name__,
        options=booster_params,
        framework_context=context,
        metadata=metadata,
    ) as ctx:  # type: StoreCtx
        ctx.options["sklearn_api"] = False
        if any(
            isinstance(model, _)
            for _ in [
                lgb.LGBMModel,
                lgb.LGBMClassifier,
                lgb.LGBMRegressor,
                lgb.LGBMRanker,
            ]
        ):
            joblib.dump(model, os.path.join(ctx.path, f"{SAVE_NAMESPACE}{PKL_EXT}"))
            ctx.options["sklearn_api"] = True
        else:
            model.save_model(os.path.join(ctx.path, f"{SAVE_NAMESPACE}{TXT_EXT}"))
        return ctx.tag


class _LightGBMRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        infer_api_callback: str,
        booster_params: t.Optional[t.Dict[str, t.Union[str, int]]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        model_info, model_file, booster_params = _get_model_info(
            tag, booster_params, model_store
        )

        self._model_store = model_store
        self._model_info = model_info
        self._model_file = model_file
        self._booster_params = booster_params
        self._infer_api_callback = infer_api_callback

    def _is_gpu(self):
        try:
            return "gpu" in self._booster_params["device"]
        except KeyError:
            return False

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        if self._is_gpu() and self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if self._is_gpu() and self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = load(
            tag=self.name,
            booster_params=self._booster_params,
            model_store=self._model_store,
        )
        self._predict_fn = getattr(self._model, self._infer_api_callback)

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: "np.ndarray") -> "np.ndarray":  # type: ignore[override]
        return self._predict_fn(input_data)


@inject
def load_runner(
    tag: str,
    infer_api_callback: str = "predict",
    *,
    booster_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_LightGBMRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.lightgbm.load_runner` implements a Runner class that
    wrap around a Lightgbm booster model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore.
        infer_api_callback (`str`, `optional`, default to `predict`):
            Inference API callback from given model. If not specified, BentoML will use default `predict`.
             Users can also choose to use `predict_proba` for supported model.
        booster_params (`t.Dict[str, t.Union[str, int]]`, default to `None`):
            Parameters for boosters. Refers to https://lightgbm.readthedocs.io/en/latest/Parameters.html
            for more information.
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.lightgbm` model

    Examples:
        import bentoml.lightgbm

        runner = bentoml.lightgbm.load_runner("my_lightgbm_model:latest")
        runner.run_batch(X_test, num_iteration=gbm.best_iteration)
    """  # noqa
    return _LightGBMRunner(
        tag=tag,
        infer_api_callback=infer_api_callback,
        booster_params=booster_params,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
