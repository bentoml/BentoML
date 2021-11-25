

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

"""Dask extensions for distributed training. See
https://xgboost.readthedocs.io/en/latest/tutorials/dask.html for simple
tutorial.  Also xgboost/demo/dask for some examples.

There are two sets of APIs in this module, one is the functional API including
``train`` and ``predict`` methods.  Another is stateful Scikit-Learner wrapper
inherited from single-node Scikit-Learn interface.

The implementation is heavily influenced by dask_xgboost:
https://github.com/dask/dask-xgboost

"""
if TYPE_CHECKING:
    ...
else:
    ...
_DaskCollection = ...
LOGGER = ...
class RabitContext:
    '''A context controling rabit initialization and finalization.'''
    def __init__(self, args: List[bytes]) -> None:
        ...
    
    def __enter__(self) -> None:
        ...
    
    def __exit__(self, *args: List) -> None:
        ...
    


def concat(value: Any) -> Any:
    '''To be replaced with dask builtin.'''
    ...

class DaskDMatrix:
    '''DMatrix holding on references to Dask DataFrame or Dask Array.  Constructing a
    `DaskDMatrix` forces all lazy computation to be carried out.  Wait for the input data
    explicitly if you want to see actual computation of constructing `DaskDMatrix`.

    See doc for :py:obj:`xgboost.DMatrix` constructor for other parameters.  DaskDMatrix
    accepts only dask collection.

    .. note::

        DaskDMatrix does not repartition or move data between workers.  It's
        the caller's responsibility to balance the data.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from dask
        if it's set to None.

    '''
    @_deprecate_positional_args
    def __init__(self, client: distributed.Client, data: _DaskCollection, label: Optional[_DaskCollection] = ..., *, weight: Optional[_DaskCollection] = ..., base_margin: Optional[_DaskCollection] = ..., missing: float = ..., silent: bool = ..., feature_names: Optional[Union[str, List[str]]] = ..., feature_types: Optional[Union[Any, List[Any]]] = ..., group: Optional[_DaskCollection] = ..., qid: Optional[_DaskCollection] = ..., label_lower_bound: Optional[_DaskCollection] = ..., label_upper_bound: Optional[_DaskCollection] = ..., feature_weights: Optional[_DaskCollection] = ..., enable_categorical: bool = ...) -> None:
        ...
    
    def __await__(self) -> Generator:
        ...
    
    def num_col(self) -> int:
        ...
    


_DataParts = List[Tuple[Any, Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]]
class DaskPartitionIter(DataIter):
    """A data iterator for `DaskDeviceQuantileDMatrix`."""
    def __init__(self, data: Tuple[Any, ...], label: Optional[Tuple[Any, ...]] = ..., weight: Optional[Tuple[Any, ...]] = ..., base_margin: Optional[Tuple[Any, ...]] = ..., qid: Optional[Tuple[Any, ...]] = ..., label_lower_bound: Optional[Tuple[Any, ...]] = ..., label_upper_bound: Optional[Tuple[Any, ...]] = ..., feature_names: Optional[Union[str, List[str]]] = ..., feature_types: Optional[Union[Any, List[Any]]] = ...) -> None:
        ...
    
    def data(self) -> Any:
        '''Utility function for obtaining current batch of data.'''
        ...
    
    def labels(self) -> Any:
        '''Utility function for obtaining current batch of label.'''
        ...
    
    def weights(self) -> Any:
        '''Utility function for obtaining current batch of label.'''
        ...
    
    def qids(self) -> Any:
        '''Utility function for obtaining current batch of query id.'''
        ...
    
    def base_margins(self) -> Any:
        '''Utility function for obtaining current batch of base_margin.'''
        ...
    
    def label_lower_bounds(self) -> Any:
        '''Utility function for obtaining current batch of label_lower_bound.
        '''
        ...
    
    def label_upper_bounds(self) -> Any:
        '''Utility function for obtaining current batch of label_upper_bound.
        '''
        ...
    
    def reset(self) -> None:
        '''Reset the iterator'''
        ...
    
    def next(self, input_data: Callable) -> int:
        '''Yield next batch of data'''
        ...
    


class DaskDeviceQuantileDMatrix(DaskDMatrix):
    '''Specialized data type for `gpu_hist` tree method.  This class is used to reduce the
    memory usage by eliminating data copies.  Internally the all partitions/chunks of data
    are merged by weighted GK sketching.  So the number of partitions from dask may affect
    training accuracy as GK generates bounded error for each merge.  See doc string for
    :py:obj:`xgboost.DeviceQuantileDMatrix` and :py:obj:`xgboost.DMatrix` for other
    parameters.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    max_bin : Number of bins for histogram construction.

    '''
    @_deprecate_positional_args
    def __init__(self, client: distributed.Client, data: _DaskCollection, label: Optional[_DaskCollection] = ..., *, weight: Optional[_DaskCollection] = ..., base_margin: Optional[_DaskCollection] = ..., missing: float = ..., silent: bool = ..., feature_names: Optional[Union[str, List[str]]] = ..., feature_types: Optional[Union[Any, List[Any]]] = ..., max_bin: int = ..., group: Optional[_DaskCollection] = ..., qid: Optional[_DaskCollection] = ..., label_lower_bound: Optional[_DaskCollection] = ..., label_upper_bound: Optional[_DaskCollection] = ..., feature_weights: Optional[_DaskCollection] = ..., enable_categorical: bool = ...) -> None:
        ...
    


def train(client: distributed.Client, params: Dict[str, Any], dtrain: DaskDMatrix, num_boost_round: int = ..., evals: Optional[List[Tuple[DaskDMatrix, str]]] = ..., obj: Optional[Objective] = ..., feval: Optional[Metric] = ..., early_stopping_rounds: Optional[int] = ..., xgb_model: Optional[Booster] = ..., verbose_eval: Union[int, bool] = ..., callbacks: Optional[List[TrainingCallback]] = ...) -> Any:
    """Train XGBoost model.

    .. versionadded:: 1.0.0

    .. note::

        Other parameters are the same as :py:func:`xgboost.train` except for
        `evals_result`, which is returned as part of function return value instead of
        argument.

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from dask
        if it's set to None.

    Returns
    -------
    results: dict
        A dictionary containing trained booster and evaluation history.  `history` field
        is the same as `eval_result` from `xgboost.train`.

        .. code-block:: python

            {'booster': xgboost.Booster,
             'history': {'train': {'logloss': ['0.48253', '0.35953']},
                         'eval': {'logloss': ['0.480385', '0.357756']}}}

    """
    ...

def predict(client: distributed.Client, model: Union[TrainReturnT, Booster, distributed.Future], data: Union[DaskDMatrix, _DaskCollection], output_margin: bool = ..., missing: float = ..., pred_leaf: bool = ..., pred_contribs: bool = ..., approx_contribs: bool = ..., pred_interactions: bool = ..., validate_features: bool = ..., iteration_range: Tuple[int, int] = ..., strict_shape: bool = ...) -> Any:
    '''Run prediction with a trained booster.

    .. note::

        Using ``inplace_predict`` might be faster when some features are not needed.  See
        :py:meth:`xgboost.Booster.predict` for details on various parameters.  When output
        has more than 2 dimensions (shap value, leaf with strict_shape), input should be
        ``da.Array`` or ``DaskDMatrix``.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client:
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model:
        The trained model.  It can be a distributed.Future so user can
        pre-scatter it onto all workers.
    data:
        Input data used for prediction.  When input is a dataframe object,
        prediction output is a series.
    missing:
        Used when input data is not DaskDMatrix.  Specify the value
        considered as missing.

    Returns
    -------
    prediction: dask.array.Array/dask.dataframe.Series
        When input data is ``dask.array.Array`` or ``DaskDMatrix``, the return value is an
        array, when input data is ``dask.dataframe.DataFrame``, return value can be
        ``dask.dataframe.Series``, ``dask.dataframe.DataFrame``, depending on the output
        shape.

    '''
    ...

def inplace_predict(client: distributed.Client, model: Union[TrainReturnT, Booster, distributed.Future], data: _DaskCollection, iteration_range: Tuple[int, int] = ..., predict_type: str = ..., missing: float = ..., validate_features: bool = ..., base_margin: Optional[_DaskCollection] = ..., strict_shape: bool = ...) -> Any:
    """Inplace prediction. See doc in :py:meth:`xgboost.Booster.inplace_predict` for details.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    client:
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model:
        See :py:func:`xgboost.dask.predict` for details.
    data :
        dask collection.
    iteration_range:
        See :py:meth:`xgboost.Booster.predict` for details.
    predict_type:
        See :py:meth:`xgboost.Booster.inplace_predict` for details.
    missing:
        Value in the input data which needs to be present as a missing
        value. If None, defaults to np.nan.
    base_margin:
        See :py:obj:`xgboost.DMatrix` for details. Right now classifier is not well
        supported with base_margin as it requires the size of base margin to be `n_classes
        * n_samples`.

        .. versionadded:: 1.4.0

    strict_shape:
        See :py:meth:`xgboost.Booster.predict` for details.

        .. versionadded:: 1.4.0

    Returns
    -------
    prediction :
        When input data is ``dask.array.Array``, the return value is an array, when input
        data is ``dask.dataframe.DataFrame``, return value can be
        ``dask.dataframe.Series``, ``dask.dataframe.DataFrame``, depending on the output
        shape.

    """
    ...

class DaskScikitLearnBase(XGBModel):
    """Base class for implementing scikit-learn interface with Dask"""
    _client = ...
    def predict(self, X: _DaskCollection, output_margin: bool = ..., ntree_limit: Optional[int] = ..., validate_features: bool = ..., base_margin: Optional[_DaskCollection] = ..., iteration_range: Optional[Tuple[int, int]] = ...) -> Any:
        ...
    
    def apply(self, X: _DaskCollection, ntree_limit: Optional[int] = ..., iteration_range: Optional[Tuple[int, int]] = ...) -> Any:
        ...
    
    def __await__(self) -> Awaitable[Any]:
        ...
    
    def __getstate__(self) -> Dict:
        ...
    
    @property
    def client(self) -> distributed.Client:
        """The dask client used in this model.  The `Client` object can not be serialized for
        transmission, so if task is launched from a worker instead of directly from the
        client process, this attribute needs to be set at that worker.

        """
        ...
    
    @client.setter
    def client(self, clt: distributed.Client) -> None:
        ...
    


@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost.""", ["estimators", "model"])
class DaskXGBRegressor(DaskScikitLearnBase, XGBRegressorBase):
    @_deprecate_positional_args
    def fit(self, X: _DaskCollection, y: _DaskCollection, *, sample_weight: Optional[_DaskCollection] = ..., base_margin: Optional[_DaskCollection] = ..., eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ..., eval_metric: Optional[Union[str, List[str], Metric]] = ..., early_stopping_rounds: Optional[int] = ..., verbose: bool = ..., xgb_model: Optional[Union[Booster, XGBModel]] = ..., sample_weight_eval_set: Optional[List[_DaskCollection]] = ..., base_margin_eval_set: Optional[List[_DaskCollection]] = ..., feature_weights: Optional[_DaskCollection] = ..., callbacks: Optional[List[TrainingCallback]] = ...) -> DaskXGBRegressor:
        ...
    


@xgboost_model_doc('Implementation of the scikit-learn API for XGBoost classification.', ['estimators', 'model'])
class DaskXGBClassifier(DaskScikitLearnBase, XGBClassifierBase):
    def fit(self, X: _DaskCollection, y: _DaskCollection, *, sample_weight: Optional[_DaskCollection] = ..., base_margin: Optional[_DaskCollection] = ..., eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ..., eval_metric: Optional[Union[str, List[str], Metric]] = ..., early_stopping_rounds: Optional[int] = ..., verbose: bool = ..., xgb_model: Optional[Union[Booster, XGBModel]] = ..., sample_weight_eval_set: Optional[List[_DaskCollection]] = ..., base_margin_eval_set: Optional[List[_DaskCollection]] = ..., feature_weights: Optional[_DaskCollection] = ..., callbacks: Optional[List[TrainingCallback]] = ...) -> DaskXGBClassifier:
        ...
    
    def predict_proba(self, X: _DaskCollection, ntree_limit: Optional[int] = ..., validate_features: bool = ..., base_margin: Optional[_DaskCollection] = ..., iteration_range: Optional[Tuple[int, int]] = ...) -> Any:
        ...
    


@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost Ranking.

    .. versionadded:: 1.4.0

""", ["estimators", "model"], end_note="""
        Note
        ----
        For dask implementation, group is not supported, use qid instead.
""")
class DaskXGBRanker(DaskScikitLearnBase, XGBRankerMixIn):
    @_deprecate_positional_args
    def __init__(self, *, objective: str = ..., **kwargs: Any) -> None:
        ...
    
    @_deprecate_positional_args
    def fit(self, X: _DaskCollection, y: _DaskCollection, *, group: Optional[_DaskCollection] = ..., qid: Optional[_DaskCollection] = ..., sample_weight: Optional[_DaskCollection] = ..., base_margin: Optional[_DaskCollection] = ..., eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = ..., eval_group: Optional[List[_DaskCollection]] = ..., eval_qid: Optional[List[_DaskCollection]] = ..., eval_metric: Optional[Union[str, List[str], Metric]] = ..., early_stopping_rounds: int = ..., verbose: bool = ..., xgb_model: Optional[Union[XGBModel, Booster]] = ..., sample_weight_eval_set: Optional[List[_DaskCollection]] = ..., base_margin_eval_set: Optional[List[_DaskCollection]] = ..., feature_weights: Optional[_DaskCollection] = ..., callbacks: Optional[List[TrainingCallback]] = ...) -> DaskXGBRanker:
        ...
    


@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost Random Forest Regressor.

    .. versionadded:: 1.4.0

""", ["model", "objective"], extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
""")
class DaskXGBRFRegressor(DaskXGBRegressor):
    @_deprecate_positional_args
    def __init__(self, *, learning_rate: Optional[float] = ..., subsample: Optional[float] = ..., colsample_bynode: Optional[float] = ..., reg_lambda: Optional[float] = ..., **kwargs: Any) -> None:
        ...
    
    def get_xgb_params(self) -> Dict[str, Any]:
        ...
    
    def get_num_boosting_rounds(self) -> int:
        ...
    


@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost Random Forest Classifier.

    .. versionadded:: 1.4.0

""", ["model", "objective"], extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
""")
class DaskXGBRFClassifier(DaskXGBClassifier):
    @_deprecate_positional_args
    def __init__(self, *, learning_rate: Optional[float] = ..., subsample: Optional[float] = ..., colsample_bynode: Optional[float] = ..., reg_lambda: Optional[float] = ..., **kwargs: Any) -> None:
        ...
    
    def get_xgb_params(self) -> Dict[str, Any]:
        ...
    
    def get_num_boosting_rounds(self) -> int:
        ...
    


