import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from .callback import TrainingCallback
from .compat import XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .core import Booster, Metric, _deprecate_positional_args

array_like = Any

class XGBRankerMixIn:
    _estimator_type = ...

_SklObjective = Optional[
    Union[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]
]
__estimator_doc = ...
__model_doc = ...
__custom_obj_note = ...

def xgboost_model_doc(
    header: str,
    items: List[str],
    extra_parameters: Optional[str] = ...,
    end_note: Optional[str] = ...,
) -> Callable[[Type], Type]: ...
@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost.""",
    ["estimators", "model", "objective"],
)
class XGBModel(XGBModelBase):
    def __init__(
        self,
        max_depth: Optional[int] = ...,
        learning_rate: Optional[float] = ...,
        n_estimators: int = ...,
        verbosity: Optional[int] = ...,
        objective: _SklObjective = ...,
        booster: Optional[str] = ...,
        tree_method: Optional[str] = ...,
        n_jobs: Optional[int] = ...,
        gamma: Optional[float] = ...,
        min_child_weight: Optional[float] = ...,
        max_delta_step: Optional[float] = ...,
        subsample: Optional[float] = ...,
        colsample_bytree: Optional[float] = ...,
        colsample_bylevel: Optional[float] = ...,
        colsample_bynode: Optional[float] = ...,
        reg_alpha: Optional[float] = ...,
        reg_lambda: Optional[float] = ...,
        scale_pos_weight: Optional[float] = ...,
        base_score: Optional[float] = ...,
        random_state: Optional[Union[np.random.RandomState, int]] = ...,
        missing: float = ...,
        num_parallel_tree: Optional[int] = ...,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = ...,
        interaction_constraints: Optional[Union[str, List[Tuple[str]]]] = ...,
        importance_type: Optional[str] = ...,
        gpu_id: Optional[int] = ...,
        validate_parameters: Optional[bool] = ...,
        predictor: Optional[str] = ...,
        enable_categorical: bool = ...,
        **kwargs: Any
    ) -> None: ...
    def __sklearn_is_fitted__(self) -> bool: ...
    def get_booster(self) -> Booster: ...
    def set_params(self, **params: Any) -> XGBModel: ...
    def get_params(self, deep: bool = ...) -> Dict[str, Any]: ...
    def get_xgb_params(self) -> Dict[str, Any]: ...
    def get_num_boosting_rounds(self) -> int: ...
    def save_model(self, fname: Union[str, os.PathLike]) -> None: ...
    def load_model(self, fname: Union[str, bytearray, os.PathLike]) -> None: ...
    @_deprecate_positional_args
    def fit(
        self,
        X: array_like,
        y: array_like,
        *,
        sample_weight: Optional[array_like] = ...,
        base_margin: Optional[array_like] = ...,
        eval_set: Optional[List[Tuple[array_like, array_like]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: Optional[bool] = ...,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[array_like]] = ...,
        base_margin_eval_set: Optional[List[array_like]] = ...,
        feature_weights: Optional[array_like] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> XGBModel: ...
    def predict(
        self,
        X: array_like,
        output_margin: bool = ...,
        ntree_limit: Optional[int] = ...,
        validate_features: bool = ...,
        base_margin: Optional[array_like] = ...,
        iteration_range: Optional[Tuple[int, int]] = ...,
    ) -> np.ndarray: ...
    def apply(
        self,
        X: array_like,
        ntree_limit: int = ...,
        iteration_range: Optional[Tuple[int, int]] = ...,
    ) -> np.ndarray: ...
    def evals_result(self) -> TrainingCallback.EvalsLog: ...
    @property
    def n_features_in_(self) -> int: ...
    @property
    def best_score(self) -> float: ...
    @property
    def best_iteration(self) -> int: ...
    @property
    def best_ntree_limit(self) -> int: ...
    @property
    def feature_importances_(self) -> np.ndarray: ...
    @property
    def coef_(self) -> np.ndarray: ...
    @property
    def intercept_(self) -> np.ndarray: ...

PredtT = ...

@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost classification.",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : int
        Number of boosting rounds.
    use_label_encoder : bool
        (Deprecated) Use the label encoder from scikit-learn to encode the labels. For new
        code, we recommend that you set this parameter to False.
""",
)
class XGBClassifier(XGBModel, XGBClassifierBase):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        objective: _SklObjective = ...,
        use_label_encoder: bool = ...,
        **kwargs: Any
    ) -> None: ...
    @_deprecate_positional_args
    def fit(
        self,
        X: array_like,
        y: array_like,
        *,
        sample_weight: Optional[array_like] = ...,
        base_margin: Optional[array_like] = ...,
        eval_set: Optional[List[Tuple[array_like, array_like]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: Optional[bool] = ...,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[array_like]] = ...,
        base_margin_eval_set: Optional[List[array_like]] = ...,
        feature_weights: Optional[array_like] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> XGBClassifier: ...
    def predict(
        self,
        X: array_like,
        output_margin: bool = ...,
        ntree_limit: Optional[int] = ...,
        validate_features: bool = ...,
        base_margin: Optional[array_like] = ...,
        iteration_range: Optional[Tuple[int, int]] = ...,
    ) -> np.ndarray: ...
    def predict_proba(
        self,
        X: array_like,
        ntree_limit: Optional[int] = ...,
        validate_features: bool = ...,
        base_margin: Optional[array_like] = ...,
        iteration_range: Optional[Tuple[int, int]] = ...,
    ) -> np.ndarray: ...
    def evals_result(self) -> TrainingCallback.EvalsLog: ...

@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest classification.",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
    use_label_encoder : bool
        (Deprecated) Use the label encoder from scikit-learn to encode the labels. For new
        code, we recommend that you set this parameter to False.
""",
)
class XGBRFClassifier(XGBClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: float = ...,
        subsample: float = ...,
        colsample_bynode: float = ...,
        reg_lambda: float = ...,
        use_label_encoder: bool = ...,
        **kwargs: Any
    ) -> None: ...
    def get_xgb_params(self) -> Dict[str, Any]: ...
    def get_num_boosting_rounds(self) -> int: ...
    @_deprecate_positional_args
    def fit(
        self,
        X: array_like,
        y: array_like,
        *,
        sample_weight: Optional[array_like] = ...,
        base_margin: Optional[array_like] = ...,
        eval_set: Optional[List[Tuple[array_like, array_like]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: Optional[bool] = ...,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[array_like]] = ...,
        base_margin_eval_set: Optional[List[array_like]] = ...,
        feature_weights: Optional[array_like] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> XGBRFClassifier: ...

@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost regression.",
    ["estimators", "model", "objective"],
)
class XGBRegressor(XGBModel, XGBRegressorBase):
    @_deprecate_positional_args
    def __init__(self, *, objective: _SklObjective = ..., **kwargs: Any) -> None: ...

@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest regression.",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
""",
)
class XGBRFRegressor(XGBRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: float = ...,
        subsample: float = ...,
        colsample_bynode: float = ...,
        reg_lambda: float = ...,
        **kwargs: Any
    ) -> None: ...
    def get_xgb_params(self) -> Dict[str, Any]: ...
    def get_num_boosting_rounds(self) -> int: ...
    @_deprecate_positional_args
    def fit(
        self,
        X: array_like,
        y: array_like,
        *,
        sample_weight: Optional[array_like] = ...,
        base_margin: Optional[array_like] = ...,
        eval_set: Optional[List[Tuple[array_like, array_like]]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: Optional[bool] = ...,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[array_like]] = ...,
        base_margin_eval_set: Optional[List[array_like]] = ...,
        feature_weights: Optional[array_like] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> XGBRFRegressor: ...

@xgboost_model_doc(
    "Implementation of the Scikit-Learn API for XGBoost Ranking.",
    ["estimators", "model"],
    end_note="""
        Note
        ----
        A custom objective function is currently not supported by XGBRanker.
        Likewise, a custom metric function is not supported either.
        Note
        ----
        Query group information is required for ranking tasks by either using the `group`
        parameter or `qid` parameter in `fit` method.
        Before fitting the model, your data need to be sorted by query group. When fitting
        the model, you need to provide an additional array that contains the size of each
        query group.
        For example, if your original data look like:
        +-------+-----------+---------------+
        |   qid |   label   |   features    |
        +-------+-----------+---------------+
        |   1   |   0       |   x_1         |
        +-------+-----------+---------------+
        |   1   |   1       |   x_2         |
        +-------+-----------+---------------+
        |   1   |   0       |   x_3         |
        +-------+-----------+---------------+
        |   2   |   0       |   x_4         |
        +-------+-----------+---------------+
        |   2   |   1       |   x_5         |
        +-------+-----------+---------------+
        |   2   |   1       |   x_6         |
        +-------+-----------+---------------+
        |   2   |   1       |   x_7         |
        +-------+-----------+---------------+
        then your group array should be ``[3, 4]``.  Sometimes using query id (`qid`)
        instead of group can be more convenient.
""",
)
class XGBRanker(XGBModel, XGBRankerMixIn):
    @_deprecate_positional_args
    def __init__(self, *, objective: str = ..., **kwargs: Any) -> None: ...
    @_deprecate_positional_args
    def fit(
        self,
        X: array_like,
        y: array_like,
        *,
        group: Optional[array_like] = ...,
        qid: Optional[array_like] = ...,
        sample_weight: Optional[array_like] = ...,
        base_margin: Optional[array_like] = ...,
        eval_set: Optional[List[Tuple[array_like, array_like]]] = ...,
        eval_group: Optional[List[array_like]] = ...,
        eval_qid: Optional[List[array_like]] = ...,
        eval_metric: Optional[Union[str, List[str], Metric]] = ...,
        early_stopping_rounds: Optional[int] = ...,
        verbose: Optional[bool] = ...,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = ...,
        sample_weight_eval_set: Optional[List[array_like]] = ...,
        base_margin_eval_set: Optional[List[array_like]] = ...,
        feature_weights: Optional[array_like] = ...,
        callbacks: Optional[List[TrainingCallback]] = ...
    ) -> XGBRanker: ...
