import typing as t
from typing import TYPE_CHECKING

import numpy as np

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

try:
    import catboost as cbt
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """catboost is required in order to use module `bentoml.catboost`, install
        catboost with `pip install catboost`. For more information, refers to
        https://catboost.ai/docs/concepts/python-installation.html
        """
    )

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..models import ModelStore

    CatBoostModel = t.Union[
        cbt.core.CatBoost,
        cbt.core.CatBoostClassifier,
        cbt.core.CatBoostRegressor,
    ]

MODULE_NAME = "bentoml.catboost"

# TODO: support cbt.Pool runner io container

CATBOOST_EXT = "cbm"


def _get_model_info(
    tag: Tag,
    model_params: t.Optional[t.Dict[str, t.Union[str, int]]],
) -> t.Tuple["Model", str, t.Dict[str, t.Any]]:
    model = bentoml.models.get(tag)
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
    model_file: str,
    model_params: t.Optional[t.Dict[str, t.Union[str, int]]],
) -> "CatBoostModel":
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

    _m: "CatBoostModel" = model.load_model(model_file)
    return _m


def load(
    tag: t.Union[str, Tag],
    model_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
) -> "CatBoostModel":
    """
    Load a CatBoost model with the given tag from the local BentoML model store.
    Args:
        tag (``str`` ``|`` :obj:`~bentoml.Tag`):
            The tag of the model to get from the store.
        model_params (:code:`Dict[str, Union[str, Any]]`, `optional`, default to :code:`None`): The parameters for loading the CatBoost model. model_type (``str``): can be one of the following:
                * ``"classifier"`` (``CatBoostClassifier``)
                * ``"regressor"`` (``CatBoostRegressor``)
    Returns:
        :obj:`catboost.core.CatBoost` ``|`` :obj:`catboost.core.CatBoostClassifier` ``|`` :obj:`catboost.core.CatBoostRegressor`:
            The CatBoost model loaded from the model store.
    Example:
    .. code-block:: python
        import bentoml
        booster = bentoml.catboost.load("my_model:latest", model_params=dict(model_type="classifier"))
    """  # noqa

    _, _model_file, _model_params = _get_model_info(tag, model_params)

    return _load_helper(_model_file, _model_params)


def save(
    name: str,
    model: "CatBoostModel",
    *,
    model_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]] = None,
    model_export_parameters: t.Optional[t.Dict[str, t.Any]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML model store.
    Args:
        name (``str``):
            The name for a given model instance. This must be a valid
            :obj:`~bentoml.Tag` name.
        model (:obj:`catboost.core.CatBoost` ``|`` :obj:`catboost.core.CatBoostClassifier` ``|`` :obj:`catboost.core.CatBoostRegressor`):
            The CatBoost model to be saved.
        model_params (``dict[str, Any]``, ``optional``, default ``None``):
            The parameters for the CatBoost model. model_type (``str``): can be one of the following:
            * ``"classifier"`` (``CatBoostClassifier``)
            * ``"regressor"``(``CatBoostRegressor``)
        model_export_parameters (:code:`dict[str, Any]`, `optional`, default to :code:`None`):
            CatBoost export parameters for the given model. See
            `the documentation <https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model#export_parameters>`_
            for CatBoost's :code:`save_model`.
        labels (``dict[str, str]``, `optional`, default ``None``):
            A dict of immutable default labels for use in model-management programs such as Yatai,
            e.g., team=nlp, stage=dev
        custom_objects (``dict[str, Any]``, `optional`, default ``None``):
            Additional Python objects to be saved alongside the model,
            e.g., a tokenizer instance, preprocessor function, model configuration json
        metadata (``dict[str, Any]``, `optional`,  default ``None``):
            User-defined metadata for storing model training context information or model evaluation
            metrics, e.g., dataset version, training parameters, or model scores
    Returns:
        :obj:`~bentoml.Tag`: A tag that can be used to access the model from the BentoML model store.
    Example:
    .. code-block:: python
        # create and train model
        model = cbt.CatBoostClassifier(...)
        model.fit(X, y)
        ...
        tag = bentoml.catboost.save("my_catboost_model", model, model_params=dict(model_type="classifier"))
    """  # noqa
    if not model_params:
        model_params = {}

    if "model_type" not in model_params:
        model_params["model_type"] = "classifier"

    context = {
        "framework_name": "catboost",
        "pip_dependencies": [f"catboost=={get_pkg_version('catboost')}"],
    }

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        options=model_params,
        metadata=metadata,
        context=context,
    ) as _model:
        path = _model.path_of(f"{SAVE_NAMESPACE}.{CATBOOST_EXT}")
        format_ = CATBOOST_EXT
        model.save_model(
            path,
            format=format_,
            export_parameters=model_export_parameters,
        )

        return _model.tag


class _CatBoostRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        model_params: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
        name: t.Optional[str] = None,
    ):
        super().__init__(tag=tag, name=name)
        self._predict_fn_name = predict_fn_name
        self._model_params = model_params
        self._model: "CatBoostModel"
        self._predict_fn: t.Any = None

    @property
    def num_replica(self) -> int:
        return int(round(self.resource_quota.cpu))

    def _setup(self) -> None:
        self._model = load(self._tag, model_params=self._model_params)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    def _run_batch(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: t.Union["ext.NpNDArray", "ext.PdDataFrame", cbt.Pool],
    ) -> "ext.NpNDArray":
        res = self._predict_fn(inputs)
        return t.cast("ext.NpNDArray", np.asarray(res))


def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "predict",
    *,
    model_params: t.Union[None, t.Dict[str, t.Union[str, int]]] = None,
    name: t.Optional[str] = None,
) -> "_CatBoostRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.catboost.load_runner` implements a Runner class that
    wrap around a CatBoost model, which optimize it for the BentoML runtime.
    Args:
        tag (``str`` ``|`` :obj:`~bentoml.Tag`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (``str``, default to :code:`predict`):
            Options for inference functions. `predict` are the default function.
        model_params (``dict[str, Any]``, `optional`, default to :code:`None`): Parameters for
            a CatBoost model. Following parameters can be specified:
                - model_type(:code:`str`): :obj:`classifier` (`CatBoostClassifier`) or :obj:`regressor` (`CatBoostRegressor`)
    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.catboost` model
    Examples:
    .. code-block:: python
        import catboost as cbt
        import pandas as pd
        input_data = pd.read_csv("/path/to/csv")
        runner = bentoml.catboost.load_runner("my_model:latest"")
        runner.run(cbt.Pool(input_data))
    """
    return _CatBoostRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        model_params=model_params,
        name=name,
    )
