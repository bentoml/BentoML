import os
import typing as t
from typing import TYPE_CHECKING

import joblib
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PKL_EXT
from ..models import TXT_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import numpy as np

    from ..models import ModelStore

try:
    import lightgbm as lgb  # noqa: F811
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """lightgbm is required in order to use module `bentoml.lightgbm`, install
        lightgbm with `pip install lightgbm`. For more information, refer to
        https://github.com/microsoft/LightGBM/tree/master/python-package
        """
    )

MODULE_NAME = "bentoml.lightgbm"


_LightGBMModelType = t.TypeVar(
    "_LightGBMModelType",
    bound=t.Union[
        "lgb.LGBMModel", "lgb.LGBMClassifier", "lgb.LGBMRegressor", "lgb.LGBMRanker"
    ],
)


def _get_model_info(
    tag: t.Union[str, Tag],
    booster_params: t.Optional[t.Dict[str, t.Union[str, int]]],
    model_store: "ModelStore",
) -> t.Tuple["Model", str, t.Dict[str, t.Any]]:
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    _fname = (
        f"{SAVE_NAMESPACE}{TXT_EXT}"
        if not model.info.options["sklearn_api"]
        else f"{SAVE_NAMESPACE}{PKL_EXT}"
    )
    model_file = model.path_of(_fname)
    _booster_params: t.Dict[str, t.Union[str, int]] = (
        dict() if not booster_params else booster_params
    )
    for key, value in model.info.options.items():
        if key not in _booster_params:
            _booster_params[key] = value

    return model, model_file, _booster_params


@inject
def load(
    tag: t.Union[str, Tag],
    booster_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Union["lgb.basic.Booster", _LightGBMModelType]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        booster_params (:code:`Dict[str, Union[str, int]]`, `optional`, defaults to `None`):
            Parameters for boosters. Refers to `Parameters Docs <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_
            for more information.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj::code:`Union[lightgbm.LGBMModel, lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, lightgbm.LGBMRanker, lightgbm.basic.Booster]`: An instance of either
        :obj::code:`Union[lightgbm.LGBMModel, lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, lightgbm.LGBMRanker]` or :obj:`lightgbm.basic.Booster`
        from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
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
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[lightgbm.basic.Booster, lightgbm.LGBMModel, lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, lightgbm.LGBMRanker]`):
            Instance of model to be saved.
        booster_params (:code:`Dict[str, Union[str, int]]`, `optional`, defaults to `None`):
            Parameters for boosters. Refers to `Parameters Doc <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_
            for more information.
        metadata (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        import lightgbm as lgb
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
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "lightgbm",
        "pip_dependencies": [f"lightgbm=={get_pkg_version('lightgbm')}"],
    }

    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=booster_params,
        context=context,
        metadata=metadata,
    )

    _model.info.options["sklearn_api"] = False
    if any(
        isinstance(model, _)
        for _ in [
            lgb.LGBMModel,
            lgb.LGBMClassifier,
            lgb.LGBMRegressor,
            lgb.LGBMRanker,
        ]
    ):
        joblib.dump(model, _model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"))
        _model.info.options["sklearn_api"] = True
    else:
        model.save_model(_model.path_of(f"{SAVE_NAMESPACE}{TXT_EXT}"))

    _model.save(model_store)

    return _model.tag


class _LightGBMRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        infer_api_callback: str,
        booster_params: t.Optional[t.Dict[str, t.Union[str, int]]],
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)
        model_info, model_file, booster_params = _get_model_info(
            tag, booster_params, model_store
        )

        self._model_store = model_store
        self._model_info = model_info
        self._model_file = model_file
        self._booster_params = booster_params
        self._infer_api_callback = infer_api_callback
        self._tag = model_store.get(tag).tag

    def _is_gpu(self):
        try:
            return "gpu" in self._booster_params["device"]
        except KeyError:
            return False

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    @property
    def num_replica(self) -> int:
        if self._is_gpu() and self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = load(
            tag=self._tag,
            booster_params=self._booster_params,
            model_store=self._model_store,
        )
        self._predict_fn = getattr(self._model, self._infer_api_callback)

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: "np.ndarray") -> "np.ndarray":  # type: ignore[override]
        return self._predict_fn(input_data)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    infer_api_callback: str = "predict",
    *,
    booster_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
    name: t.Optional[str] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_LightGBMRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.lightgbm.load_runner` implements a Runner class that
    wrap around a LightGBM model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML modelstore.
        infer_api_callback (:code:`str`, `optional`, default to :code:`predict`):
            Inference API callback from given model. If not specified, BentoML will use default :code:`predict`.
            Users can also choose to use :code:`predict_proba` for supported model.
        booster_params (:code:`Dict[str, Union[str, int]]`, `optional`, defaults to `None`):
            Parameters for boosters. Refers to `Parameters Doc <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_
            for more information.
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.lightgbm` model

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.lightgbm.load_runner("my_lightgbm_model:latest")
        runner.run_batch(X_test, num_iteration=gbm.best_iteration)
    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _LightGBMRunner(
        tag=tag,
        infer_api_callback=infer_api_callback,
        name=name,
        booster_params=booster_params,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
