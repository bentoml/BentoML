from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import distributed
from .callback import TrainingCallback
from .core import Booster, DataIter, Metric, Objective, _deprecate_positional_args
from .sklearn import (
    XGBClassifierBase,
    XGBModel,
    XGBRankerMixIn,
    XGBRegressorBase,
    xgboost_model_doc,
)

if TYPE_CHECKING: ...
else: ...
_DaskCollection = ...
__all__ = [
    "RabitContext",
    "DaskDMatrix",
    "DaskDeviceQuantileDMatrix",
    "DaskXGBRegressor",
    "DaskXGBClassifier",
    "DaskXGBRanker",
    "DaskXGBRFRegressor",
    "DaskXGBRFClassifier",
    "train",
    "predict",
    "inplace_predict",
]
LOGGER = ...

class RabitContext:
    def __init__(self, args: List[bytes]) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: List) -> None: ...

def concat(value: Any) -> Any: ...

class DaskDMatrix:
    @_deprecate_positional_args
    def __init__(
        self,
        client: distributed.Client,
        data: _DaskCollection,
        label: Optional[_DaskCollection] = ...,
        *,
        weight: Optional[_DaskCollection] = ...,
        base_margin: Optional[_DaskCollection] = ...,
        missing: float = ...,
        silent: bool = ...,
        feature_names: Optional[Union[str, List[str]]] = ...,
        feature_types: Optional[Union[Any, List[Any]]] = ...,
        group: Optional[_DaskCollection] = ...,
        qid: Optional[_DaskCollection] = ...,
        label_lower_bound: Optional[_DaskCollection] = ...,
        label_upper_bound: Optional[_DaskCollection] = ...,
        feature_weights: Optional[_DaskCollection] = ...,
        enable_categorical: bool = ...
    ) -> None: ...
    def __await__(self) -> Generator: ...
    def num_col(self) -> int: ...

_DataParts = List[
    Tuple[
        Any,
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
    ]
]

class DaskPartitionIter(DataIter):
    def __init__(
        self,
        data: Tuple[Any, ...],
        label: Optional[Tuple[Any, ...]] = ...,
        weight: Optional[Tuple[Any, ...]] = ...,
        base_margin: Optional[Tuple[Any, ...]] = ...,
        qid: Optional[Tuple[Any, ...]] = ...,
        label_lower_bound: Optional[Tuple[Any, ...]] = ...,
        label_upper_bound: Optional[Tuple[Any, ...]] = ...,
        feature_names: Optional[Union[str, List[str]]] = ...,
        feature_types: Optional[Union[Any, List[Any]]] = ...,
    ) -> None: ...
    def data(self) -> Any: ...
    def labels(self) -> Any: ...
    def weights(self) -> Any: ...
    def qids(self) -> Any: ...
    def base_margins(self) -> Any: ...
    def label_lower_bounds(self) -> Any: ...
    def label_upper_bounds(self) -> Any: ...
    def reset(self) -> None: ...
    def next(self, input_data: Callable) -> int: ...

class DaskDeviceQuantileDMatrix(DaskDMatrix):
    @_deprecate_positional_args
    def __init__(
        self,
        client: distributed.Client,
        data: _DaskCollection,
        label: Optional[_DaskCollection] = ...,
        *,
        weight: Optional[_DaskCollection] = ...,
        base_margin: Optional[_DaskCollection] = ...,
        missing: float = ...,
        silent: bool = ...,
        feature_names: Optional[Union[str, List[str]]] = ...,
        feature_types: Optional[Union[Any, List[Any]]] = ...,
        max_bin: int = ...,
        group: Optional[_DaskCollection] = ...,
        qid: Optional[_DaskCollection] = ...,
        label_lower_bound: Optional[_DaskCollection] = ...,
        label_upper_bound: Optional[_DaskCollection] = ...,
        feature_weights: Optional[_DaskCollection] = ...,
        enable_categorical: bool = ...
    ) -> None: ...

def train(
    client: distributed.Client,
    params: Dict[str, Any],
    dtrain: DaskDMatrix,
    num_boost_round: int = ...,
    evals: Optional[List[Tuple[DaskDMatrix, str]]] = ...,
    obj: Optional[Objective] = ...,
    feval: Optional[Metric] = ...,
    early_stopping_rounds: Optional[int] = ...,
    xgb_model: Optional[Booster] = ...,
    verbose_eval: Union[int, bool] = ...,
    callbacks: Optional[List[TrainingCallback]] = ...,
) -> Any: ...
def predict(
    client: distributed.Client,
    model: Union[TrainReturnT, Booster, distributed.Future],
    data: Union[DaskDMatrix, _DaskCollection],
    output_margin: bool = ...,
    missing: float = ...,
    pred_leaf: bool = ...,
    pred_contribs: bool = ...,
    approx_contribs: bool = ...,
    pred_interactions: bool = ...,
    validate_features: bool = ...,
    iteration_range: Tuple[int, int] = ...,
    strict_shape: bool = ...,
) -> Any: ...
def inplace_predict(
    client: distributed.Client,
    model: Union[TrainReturnT, Booster, distributed.Future],
    data: _DaskCollection,
    iteration_range: Tuple[int, int] = ...,
    predict_type: str = ...,
    missing: float = ...,
    validate_features: bool = ...,
    base_margin: Optional[_DaskCollection] = ...,
    strict_shape: bool = ...,
) -> Any: ...

class DaskScikitLearnBase(XGBModel):
    _client = ...
    def predict(
        self,
        X: _DaskCollection,
        output_margin: bool = ...,
        ntree_limit: Optional[int] = ...,
        validate_features: bool = ...,
        base_margin: Optional[_DaskCollection] = ...,
        iteration_range: Optional[Tuple[int, int]] = ...,
    ) -> Any: ...
    def apply(
        self,
        X: _DaskCollection,
        ntree_limit: Optional[int] = ...,
        iteration_range: Optional[Tuple[int, int]] = ...,
    ) -> Any: ...
    def __await__(self) -> Awaitable[Any]: ...
    def __getstate__(self) -> Dict: ...
    @property
    def client(self) -> distributed.Client: ...
    @client.setter
    def client(self, clt: distributed.Client) -> None: ...

@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost.""", ["estimators", "model"]
)
class DaskXGBRegressor(DaskScikitLearnBase, XGBRegressorBase):
    @_deprecate_positional_args
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = ...,
        base_margin: Optional[_DaskCollection] = ...,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: bool = ...,
        xgb_model: Optional[Union[Booster, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = ...,
        base_margin_eval_set: Optional[List[_DaskCollection]] = ...,
        feature_weights: Optional[_DaskCollection] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> DaskXGBRegressor: ...

@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost classification.",
    ["estimators", "model"],
)
class DaskXGBClassifier(DaskScikitLearnBase, XGBClassifierBase):
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = ...,
        base_margin: Optional[_DaskCollection] = ...,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: bool = ...,
        xgb_model: Optional[Union[Booster, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = ...,
        base_margin_eval_set: Optional[List[_DaskCollection]] = ...,
        feature_weights: Optional[_DaskCollection] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> DaskXGBClassifier: ...
    def predict_proba(
        self,
        X: _DaskCollection,
        ntree_limit: Optional[int] = ...,
        validate_features: bool = ...,
        base_margin: Optional[_DaskCollection] = ...,
        iteration_range: Optional[Tuple[int, int]] = ...,
    ) -> Any: ...

@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost Ranking.
    .. versionadded:: 1.4.0
""",
    ["estimators", "model"],
    end_note="""
        Note
        ----
        For dask implementation, group is not supported, use qid instead.
""",
)
class DaskXGBRanker(DaskScikitLearnBase, XGBRankerMixIn):
    @_deprecate_positional_args
    def __init__(self, *, objective: str = ..., **kwargs: Any) -> None: ...
    @_deprecate_positional_args
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        group: Optional[_DaskCollection] = ...,
        qid: Optional[_DaskCollection] = ...,
        sample_weight: Optional[_DaskCollection] = ...,
        base_margin: Optional[_DaskCollection] = ...,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ...,
        eval_group: Optional[List[_DaskCollection]] = ...,
        eval_qid: Optional[List[_DaskCollection]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: int = ...,
        verbose: bool = ...,
        xgb_model: Optional[Union[XGBModel, Booster]] = ...,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = ...,
        base_margin_eval_set: Optional[List[_DaskCollection]] = ...,
        feature_weights: Optional[_DaskCollection] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> DaskXGBRanker: ...

@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost Random Forest Regressor.
    .. versionadded:: 1.4.0
""",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
""",
)
class DaskXGBRFRegressor(DaskXGBRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: Optional[float] = ...,
        subsample: Optional[float] = ...,
        colsample_bynode: Optional[float] = ...,
        reg_lambda: Optional[float] = ...,
        **kwargs: Any
    ) -> None: ...
    def get_xgb_params(self) -> Dict[str, Any]: ...
    def get_num_boosting_rounds(self) -> int: ...
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = ...,
        base_margin: Optional[_DaskCollection] = ...,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: bool = ...,
        xgb_model: Optional[Union[Booster, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = ...,
        base_margin_eval_set: Optional[List[_DaskCollection]] = ...,
        feature_weights: Optional[_DaskCollection] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> DaskXGBRFRegressor: ...

@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost Random Forest Classifier.
    .. versionadded:: 1.4.0
""",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
""",
)
class DaskXGBRFClassifier(DaskXGBClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: Optional[float] = ...,
        subsample: Optional[float] = ...,
        colsample_bynode: Optional[float] = ...,
        reg_lambda: Optional[float] = ...,
        **kwargs: Any
    ) -> None: ...
    def get_xgb_params(self) -> Dict[str, Any]: ...
    def get_num_boosting_rounds(self) -> int: ...
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = ...,
        base_margin: Optional[_DaskCollection] = ...,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: bool = ...,
        xgb_model: Optional[Union[Booster, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = ...,
        base_margin_eval_set: Optional[List[_DaskCollection]] = ...,
        feature_weights: Optional[_DaskCollection] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> DaskXGBRFClassifier: ...
