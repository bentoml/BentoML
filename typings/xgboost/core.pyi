

import ctypes
import os
from types import MappingProxyType
from typing import (
    Any,
    ByteString,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import sparse

AnyNdarray = np.ndarray[Any, np.dtype[Any]]

"""Core XGBoost Library."""
c_bst_ulong = ctypes.c_uint64
class XGBoostError(ValueError):
    """Error thrown by xgboost trainer."""
    ...


class EarlyStopException(Exception):
    """Exception to signal early stopping.

    Parameters
    ----------
    best_iteration : int
        The best iteration stopped.
    """
    def __init__(self, best_iteration) -> None:
        ...
    


CallbackEnv = ...
def from_pystr_to_cstr(data: Union[str, List[str]]): # -> bytes | Array[c_char_p]:
    """Convert a Python str or list of Python str to C pointer

    Parameters
    ----------
    data
        str or list of str
    """
    ...

def from_cstr_to_pystr(data, length) -> List[str]:
    """Revert C pointer to Python str

    Parameters
    ----------
    data : ctypes pointer
        pointer to data
    length : ctypes pointer
        pointer to length of data
    """
    ...

_LIB = ...
def ctypes2numpy(cptr, length, dtype): # -> ndarray[Unknown, Unknown]:
    """Convert a ctypes pointer array to a numpy array."""
    ...

def ctypes2cupy(cptr, length, dtype):
    """Convert a ctypes pointer array to a cupy array."""
    ...

def ctypes2buffer(cptr, length) -> ByteString:
    """Convert ctypes pointer to buffer type."""
    ...

def c_str(string): # -> c_char_p:
    """Convert a python string to cstring."""
    ...

def c_array(ctype, values):
    """Convert a python string to c array."""
    ...

class DataIter:
    '''The interface for user defined data iterator. Currently is only supported by Device
    DMatrix.

    '''
    def __init__(self) -> None:
        ...
    
    @property
    def proxy(self): # -> _ProxyDMatrix:
        '''Handler of DMatrix proxy.'''
        ...
    
    def reset_wrapper(self, this): # -> None:
        '''A wrapper for user defined `reset` function.'''
        ...
    
    def next_wrapper(self, this): # -> Literal[0]:
        '''A wrapper for user defined `next` function.

        `this` is not used in Python.  ctypes can handle `self` of a Python
        member function automatically when converting it to c function
        pointer.

        '''
        ...
    
    def reset(self):
        '''Reset the data iterator.  Prototype for user defined function.'''
        ...
    
    def next(self, input_data):
        '''Set the next batch of data.

        Parameters
        ----------

        data_handle: callable
            A function with same data fields like `data`, `label` with
            `xgboost.DMatrix`.

        Returns
        -------
        0 if there's no more batch, otherwise 1.

        '''
        ...
    

DMatrixDataType = Union[os.PathLike[str], str, AnyNdarray, pd.DataFrame, sparse.spmatrix]

class DMatrix:
    """Data Matrix used in XGBoost.

    DMatrix is an internal data structure that is used by XGBoost,
    which is optimized for both memory efficiency and training speed.
    You can construct DMatrix from multiple different sources of data.
    """
    def __init__(self, data: DMatrixDataType, label=..., *, weight=..., base_margin=..., missing: Optional[float] = ..., silent=..., feature_names=..., feature_types=..., nthread: Optional[int] = ..., group=..., qid=..., label_lower_bound=..., label_upper_bound=..., feature_weights=..., enable_categorical: bool = ...) -> None:
        """Parameters
        ----------
        data : os.PathLike/string/numpy.array/scipy.sparse/pd.DataFrame/
               dt.Frame/cudf.DataFrame/cupy.array/dlpack
            Data source of DMatrix.
            When data is string or os.PathLike type, it represents the path
            libsvm format txt file, csv file (by specifying uri parameter
            'path_to_csv?format=csv'), or binary file that xgboost can read
            from.
        label : array_like
            Label of the training data.
        weight : array_like
            Weight for each instance.

            .. note:: For ranking task, weights are per-group.

                In ranking task, one weight is assigned to each group (not each
                data point). This is because we only care about the relative
                ordering of data points within each group, so it doesn't make
                sense to assign weights to individual data points.

        base_margin: array_like
            Base margin used for boosting from existing model.
        missing : float, optional
            Value in the input data which needs to be present as a missing
            value. If None, defaults to np.nan.
        silent : boolean, optional
            Whether print messages during construction
        feature_names : list, optional
            Set names for features.
        feature_types : list, optional
            Set types for features.
        nthread : integer, optional
            Number of threads to use for loading data when parallelization is
            applicable. If -1, uses maximum threads available on the system.
        group : array_like
            Group size for all ranking group.
        qid : array_like
            Query ID for data samples, used for ranking.
        label_lower_bound : array_like
            Lower bound for survival training.
        label_upper_bound : array_like
            Upper bound for survival training.
        feature_weights : array_like, optional
            Set feature weights for column sampling.
        enable_categorical: boolean, optional

            .. versionadded:: 1.3.0

            Experimental support of specializing for categorical features.  Do
            not set to True unless you are interested in development.
            Currently it's only available for `gpu_hist` tree method with 1 vs
            rest (one hot) categorical split.  Also, JSON serialization format,
            `gpu_predictor` and pandas input are required.

        """
        ...
    
    def __del__(self): # -> None:
        ...
    
    def set_info(self, *, label=..., weight=..., base_margin=..., group=..., qid=..., label_lower_bound=..., label_upper_bound=..., feature_names=..., feature_types=..., feature_weights=...) -> None:
        """Set meta info for DMatrix.  See doc string for :py:obj:`xgboost.DMatrix`."""
        ...
    
    def get_float_info(self, field): # -> ndarray[Unknown, Unknown]:
        """Get float property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of float information of the data
        """
        ...
    
    def get_uint_info(self, field): # -> ndarray[Unknown, Unknown]:
        """Get unsigned integer property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of unsigned integer information of the data
        """
        ...
    
    def set_float_info(self, field, data): # -> None:
        """Set float type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        ...
    
    def set_float_info_npy2d(self, field, data): # -> None:
        """Set float type property into the DMatrix
           for numpy 2d array input

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        ...
    
    def set_uint_info(self, field, data): # -> None:
        """Set uint type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        ...
    
    def save_binary(self, fname, silent=...): # -> None:
        """Save DMatrix to an XGBoost buffer.  Saved binary can be later loaded
        by providing the path to :py:func:`xgboost.DMatrix` as input.

        Parameters
        ----------
        fname : string or os.PathLike
            Name of the output buffer file.
        silent : bool (optional; default: True)
            If set, the output is suppressed.
        """
        ...
    
    def set_label(self, label): # -> None:
        """Set label of dmatrix

        Parameters
        ----------
        label: array like
            The label information to be set into DMatrix
        """
        ...
    
    def set_weight(self, weight): # -> None:
        """Set weight of each instance.

        Parameters
        ----------
        weight : array like
            Weight for each data point

            .. note:: For ranking task, weights are per-group.

                In ranking task, one weight is assigned to each group (not each
                data point). This is because we only care about the relative
                ordering of data points within each group, so it doesn't make
                sense to assign weights to individual data points.

        """
        ...
    
    def set_base_margin(self, margin): # -> None:
        """Set base margin of booster to start from.

        This can be used to specify a prediction value of existing model to be
        base_margin However, remember margin is needed, instead of transformed
        prediction e.g. for logistic regression: need to put in value before
        logistic transformation see also example/demo.py

        Parameters
        ----------
        margin: array like
            Prediction margin of each datapoint

        """
        ...
    
    def set_group(self, group): # -> None:
        """Set group size of DMatrix (used for ranking).

        Parameters
        ----------
        group : array like
            Group size of each group
        """
        ...
    
    def get_label(self): # -> ndarray[Unknown, Unknown]:
        """Get the label of the DMatrix.

        Returns
        -------
        label : array
        """
        ...
    
    def get_weight(self): # -> ndarray[Unknown, Unknown]:
        """Get the weight of the DMatrix.

        Returns
        -------
        weight : array
        """
        ...
    
    def get_base_margin(self): # -> ndarray[Unknown, Unknown]:
        """Get the base margin of the DMatrix.

        Returns
        -------
        base_margin : float
        """
        ...
    
    def num_row(self): # -> int:
        """Get the number of rows in the DMatrix.

        Returns
        -------
        number of rows : int
        """
        ...
    
    def num_col(self): # -> int:
        """Get the number of columns (features) in the DMatrix.

        Returns
        -------
        number of columns : int
        """
        ...
    
    def slice(self, rindex: Union[List[int], AnyNdarray], allow_groups: bool = ...) -> DMatrix:
        """Slice the DMatrix and return a new DMatrix that only contains `rindex`.

        Parameters
        ----------
        rindex
            List of indices to be selected.
        allow_groups
            Allow slicing of a matrix with a groups attribute

        Returns
        -------
        res
            A new DMatrix containing only selected indices.
        """
        ...
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names (column labels).

        Returns
        -------
        feature_names : list or None
        """
        ...
    
    @feature_names.setter
    def feature_names(self, feature_names: Optional[Union[List[str], str]]) -> None:
        """Set feature names (column labels).

        Parameters
        ----------
        feature_names : list or None
            Labels for features. None will reset existing feature names
        """
        ...
    
    @property
    def feature_types(self) -> Optional[List[str]]:
        """Get feature types (column types).

        Returns
        -------
        feature_types : list or None
        """
        ...
    
    @feature_types.setter
    def feature_types(self, feature_types: Optional[Union[List[Any], Any]]) -> None:
        """Set feature types (column types).

        This is for displaying the results and unrelated
        to the learning process.

        Parameters
        ----------
        feature_types : list or None
            Labels for features. None will reset existing feature names
        """
        ...
    


class _ProxyDMatrix(DMatrix):
    """A placeholder class when DMatrix cannot be constructed (DeviceQuantileDMatrix,
    inplace_predict).

    """
    def __init__(self) -> None:
        ...
    


class DeviceQuantileDMatrix(DMatrix):
    """Device memory Data Matrix used in XGBoost for training with tree_method='gpu_hist'. Do
    not use this for test/validation tasks as some information may be lost in
    quantisation. This DMatrix is primarily designed to save memory in training from
    device memory inputs by avoiding intermediate storage. Set max_bin to control the
    number of bins during quantisation.  See doc string in :py:obj:`xgboost.DMatrix` for
    documents on meta info.

    You can construct DeviceQuantileDMatrix from cupy/cudf/dlpack.

    .. versionadded:: 1.1.0

    """
    @_deprecate_positional_args
    def __init__(self, data, label=..., *, weight=..., base_margin=..., missing=..., silent=..., feature_names=..., feature_types=..., nthread: Optional[int] = ..., max_bin: int = ..., group=..., qid=..., label_lower_bound=..., label_upper_bound=..., feature_weights=..., enable_categorical: bool = ...) -> None:
        ...
    


Objective = Callable[[AnyNdarray, DMatrix], Tuple[AnyNdarray,...]]
Metric = Callable[[AnyNdarray, DMatrix], Tuple[str, float]]
class Booster:
    """A Booster of XGBoost.

    Booster is the model of xgboost, that contains low level routines for
    training, prediction and evaluation.
    """
    def __init__(self, params: Dict[str, Any]=..., cache: List[DMatrix]=..., model_file: Union[str, os.PathLike[str], "Booster", ByteString]=...) -> None:
        """
        Parameters
        ----------
        params : dict
            Parameters for boosters.
        cache : list
            List of cache items.
        model_file : string/os.PathLike/Booster/bytearray
            Path to the model file if it's string or PathLike.
        """
        ...
    def _configure_metrics(self, params: Union[Dict[str, Any], List[str]]) -> Union[Dict[str, Any], List[str]]:
        ...
    
    def __del__(self) -> None:
        ...
    
    def __getstate__(self) -> Dict[str, Any]:
        ...
    
    def __setstate__(self, state: MappingProxyType[str, Any])-> Dict[str, Any]:
        ...
    
    def __getitem__(self, val: Any) -> "Booster":
        ...
    
    def save_config(self) -> str:
        '''Output internal parameter configuration of Booster as a JSON
        string.

        .. versionadded:: 1.0.0
        '''
        ...
    
    def load_config(self, config: str) -> None:
        '''Load configuration returned by `save_config`.

        .. versionadded:: 1.0.0
        '''
        ...
    
    def __copy__(self) -> "Booster":
        ...
    
    def __deepcopy__(self, _) -> "Booster":
        '''Return a copy of booster.'''
        ...
    
    def copy(self) -> "Booster":
        """Copy the booster object.

        Returns
        -------
        booster: `Booster`
            a copied booster model
        """
        ...
    
    def attr(self, key: str) -> Optional[str]:
        """Get attribute string from the Booster.

        Parameters
        ----------
        key : str
            The key to get attribute from.

        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        ...
    
    def attributes(self) -> Dict[str, Optional[str]]:
        """Get attributes stored in the Booster as a dictionary.

        Returns
        -------
        result : dictionary of  attribute_name: attribute_value pairs of strings.
            Returns an empty dict if there's no attributes.
        """
        ...
    
    def set_attr(self, **kwargs: str) -> None:
        """Set the attribute of the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set. Setting a value to None deletes an attribute.
        """
        ...
    
    @property
    def feature_types(self) -> Optional[List[str]]:
        """Feature types for this booster.  Can be directly set by input data or by
        assignment.

        """
        ...
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        """Feature names for this booster.  Can be directly set by input data or by
        assignment.

        """
        ...
    
    @feature_names.setter
    def feature_names(self, features: Optional[List[str]]) -> None:
        ...
    
    @feature_types.setter
    def feature_types(self, features: Optional[List[str]]) -> None:
        ...
    
    def set_param(self, params: Union[Dict[str, Any], List[Mapping[str, str]], str], value: Optional[str]=...) -> None:
        """Set parameters into the Booster.

        Parameters
        ----------
        params: dict/list/str
           list of key,value pairs, dict of key to value or simply str key
        value: optional
           value of the specified parameter, when params is str key
        """
        ...
    
    def update(self, dtrain: DMatrix, iteration: int, fobj: Callable[..., Any]=...) -> None:
        """Update for one iteration, with objective function calculated
        internally.  This function should not be called directly by users.

        Parameters
        ----------
        dtrain : DMatrix
            Training data.
        iteration : int
            Current iteration number.
        fobj : function
            Customized objective function.

        """
        ...
    
    def boost(self, dtrain: DMatrix, grad: List[Any], hess: List[Any]) -> None:
        """Boost the booster for one iteration, with customized gradient
        statistics.  Like :py:func:`xgboost.Booster.update`, this
        function should not be called directly by users.

        Parameters
        ----------
        dtrain : DMatrix
            The training DMatrix.
        grad : list
            The first order of gradient.
        hess : list
            The second order of gradient.

        """
        ...
    
    def eval_set(self, evals: List[Tuple[DMatrix, str]], iteration: int=..., feval: Callable[..., Any]=...)-> str:
        """Evaluate a set of data.

        Parameters
        ----------
        evals : list of tuples (DMatrix, string)
            List of items to be evaluated.
        iteration : int
            Current iteration.
        feval : function
            Custom evaluation function.

        Returns
        -------
        result: str
            Evaluation result string.
        """
        ...
    
    def eval(self, data: DMatrix, name: str=..., iteration: int=...) -> str:
        """Evaluate the model on mat.

        Parameters
        ----------
        data : DMatrix
            The dmatrix storing the input.

        name : str, optional
            The name of the dataset.

        iteration : int, optional
            The current iteration number.

        Returns
        -------
        result: str
            Evaluation result string.
        """
        ...
    
    def predict(self, data: DMatrix, output_margin: bool = ..., ntree_limit: int = ..., pred_leaf: bool = ..., pred_contribs: bool = ..., approx_contribs: bool = ..., pred_interactions: bool = ..., validate_features: bool = ..., training: bool = ..., iteration_range: Tuple[int, int] = ..., strict_shape: bool = ...) -> AnyNdarray:
        """Predict with data.

          .. note:: This function is not thread safe except for ``gbtree`` booster.

          When using booster other than ``gbtree``, predict can only be called from one
          thread.  If you want to run prediction using multiple thread, call
          :py:meth:`xgboost.Booster.copy` to make copies of model object and then call
          ``predict()``.

        Parameters
        ----------
        data :
            The dmatrix storing the input.

        output_margin :
            Whether to output the raw untransformed margin value.

        ntree_limit :
            Deprecated, use `iteration_range` instead.

        pred_leaf :
            When this option is on, the output will be a matrix of (nsample,
            ntrees) with each record indicating the predicted leaf index of
            each sample in each tree.  Note that the leaf index of a tree is
            unique per tree, so you may find leaf 1 in both tree 1 and tree 0.

        pred_contribs :
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1) with each record indicating the feature contributions
            (SHAP values) for that prediction. The sum of all feature
            contributions is equal to the raw untransformed margin value of the
            prediction. Note the final column is the bias term.

        approx_contribs :
            Approximate the contributions of each feature.  Used when ``pred_contribs`` or
            ``pred_interactions`` is set to True.  Changing the default of this parameter
            (False) is not recommended.

        pred_interactions :
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1, nfeats + 1) indicating the SHAP interaction values for
            each pair of features. The sum of each row (or column) of the
            interaction values equals the corresponding SHAP value (from
            pred_contribs), and the sum of the entire matrix equals the raw
            untransformed margin value of the prediction. Note the last row and
            column correspond to the bias term.

        validate_features :
            When this is True, validate that the Booster's and data's
            feature_names are identical.  Otherwise, it is assumed that the
            feature_names are the same.

        training :
            Whether the prediction value is used for training.  This can effect
            `dart` booster, which performs dropouts during training iterations.

            .. versionadded:: 1.0.0

        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

            .. versionadded:: 1.4.0

        strict_shape :
            When set to True, output shape is invariant to whether classification is used.
            For both value and margin prediction, the output shape is (n_samples,
            n_groups), n_groups == 1 when multi-class is not used.  Default to False, in
            which case the output shape can be (n_samples, ) if multi-class is not used.

            .. versionadded:: 1.4.0

        .. note:: Using ``predict()`` with DART booster

          If the booster object is DART type, ``predict()`` will not perform
          dropouts, i.e. all the trees will be evaluated.  If you want to
          obtain result with dropouts, provide `training=True`.

        Returns
        -------
        prediction : numpy array

        """
        ...
    
    def inplace_predict(self, data: Any, iteration_range: Tuple[int, int] = ..., predict_type: str = ..., missing: float = ..., validate_features: bool = ..., base_margin: Any = ..., strict_shape: bool = ...): # -> ndarray[Unknown, Unknown]:
        """Run prediction in-place, Unlike ``predict`` method, inplace prediction does
        not cache the prediction result.

        Calling only ``inplace_predict`` in multiple threads is safe and lock
        free.  But the safety does not hold when used in conjunction with other
        methods. E.g. you can't train the booster in one thread and perform
        prediction in the other.

        .. code-block:: python

            booster.set_param({'predictor': 'gpu_predictor'})
            booster.inplace_predict(cupy_array)

            booster.set_param({'predictor': 'cpu_predictor})
            booster.inplace_predict(numpy_array)

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data : numpy.ndarray/scipy.sparse.csr_matrix/cupy.ndarray/
               cudf.DataFrame/pd.DataFrame
            The input data, must not be a view for numpy array.  Set
            ``predictor`` to ``gpu_predictor`` for running prediction on CuPy
            array or CuDF DataFrame.
        iteration_range :
            See :py:meth:`xgboost.Booster.predict` for details.
        predict_type :
            * `value` Output model prediction values.
            * `margin` Output the raw untransformed margin value.
        missing :
            See :py:obj:`xgboost.DMatrix` for details.
        validate_features:
            See :py:meth:`xgboost.Booster.predict` for details.
        base_margin:
            See :py:obj:`xgboost.DMatrix` for details.

            .. versionadded:: 1.4.0

        strict_shape:
            See :py:meth:`xgboost.Booster.predict` for details.

            .. versionadded:: 1.4.0

        Returns
        -------
        prediction : numpy.ndarray/cupy.ndarray
            The prediction result.  When input data is on GPU, prediction
            result is stored in a cupy array.

        """
        ...
    
    def save_model(self, fname: Union[str, os.PathLike[str]]) -> None:
        """Save the model to a file.

        The model is saved in an XGBoost internal format which is universal among the
        various XGBoost interfaces. Auxiliary attributes of the Python Booster object
        (such as feature_names) will not be saved when using binary format.  To save those
        attributes, use JSON instead. See:

          https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

        for more info.

        Parameters
        ----------
        fname : string or os.PathLike
            Output file name

        """
        ...
    
    def save_raw(self) -> ByteString:
        """Save the model to a in memory buffer representation instead of file.

        Returns
        -------
        a in memory buffer representation of the model
        """
        ...
    
    def load_model(self, fname: Union[str, os.PathLike[str], ByteString]) -> None:
        """Load the model from a file or bytearray. Path to file can be local
        or as an URI.

        The model is loaded from XGBoost format which is universal among the various
        XGBoost interfaces. Auxiliary attributes of the Python Booster object (such as
        feature_names) will not be loaded when using binary format.  To save those
        attributes, use JSON instead.  See:

          https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

        for more info.

        Parameters
        ----------
        fname : string, os.PathLike, or a memory buffer
            Input file name or memory buffer(see also save_raw)

        """
        ...
    
    def num_boosted_rounds(self) -> int:
        '''Get number of boosted rounds.  For gblinear this is reset to 0 after
        serializing the model.

        '''
        ...
    
    def num_features(self) -> int:
        '''Number of features in booster.'''
        ...
    
    def dump_model(self, fout, fmap=..., with_stats=..., dump_format=...): # -> None:
        """Dump model into a text or JSON file.  Unlike `save_model`, the
        output format is primarily used for visualization or interpretation,
        hence it's more human readable but cannot be loaded back to XGBoost.

        Parameters
        ----------
        fout : string or os.PathLike
            Output file name.
        fmap : string or os.PathLike, optional
            Name of the file containing feature map names.
        with_stats : bool, optional
            Controls whether the split statistics are output.
        dump_format : string, optional
            Format of model dump file. Can be 'text' or 'json'.
        """
        ...
    
    def get_dump(self, fmap=..., with_stats=..., dump_format=...): # -> List[str]:
        """Returns the model dump as a list of strings.  Unlike `save_model`, the
        output format is primarily used for visualization or interpretation,
        hence it's more human readable but cannot be loaded back to XGBoost.

        Parameters
        ----------
        fmap : string or os.PathLike, optional
            Name of the file containing feature map names.
        with_stats : bool, optional
            Controls whether the split statistics are output.
        dump_format : string, optional
            Format of model dump. Can be 'text', 'json' or 'dot'.

        """
        ...
    
    def get_fscore(self, fmap=...): # -> dict[Unknown, Unknown]:
        """Get feature importance of each feature.

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        .. note:: Zero-importance features will not be included

           Keep in mind that this function does not include zero-importance feature, i.e.
           those features that have not been used in any split conditions.

        Parameters
        ----------
        fmap: str or os.PathLike (optional)
           The name of feature map file
        """
        ...
    
    def get_score(self, fmap=..., importance_type=...):
        """Get feature importance of each feature.
        Importance type can be defined as:

        * 'weight': the number of times a feature is used to split the data across all trees.
        * 'gain': the average gain across all splits the feature is used in.
        * 'cover': the average coverage across all splits the feature is used in.
        * 'total_gain': the total gain across all splits the feature is used in.
        * 'total_cover': the total coverage across all splits the feature is used in.

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        Parameters
        ----------
        fmap: str or os.PathLike (optional)
           The name of feature map file.
        importance_type: str, default 'weight'
            One of the importance types defined above.
        """
        ...
    
    def trees_to_dataframe(self, fmap=...):
        """Parse a boosted tree model text dump into a pandas DataFrame structure.

        This feature is only defined when the decision tree model is chosen as base
        learner (`booster in {gbtree, dart}`). It is not defined for other base learner
        types, such as linear learners (`booster=gblinear`).

        Parameters
        ----------
        fmap: str or os.PathLike (optional)
           The name of feature map file.
        """
        ...
    
    def get_split_value_histogram(self, feature, fmap=..., bins=..., as_pandas=...): # -> object:
        """Get split value histogram of a feature

        Parameters
        ----------
        feature: str
            The name of the feature.
        fmap: str or os.PathLike (optional)
            The name of feature map file.
        bin: int, default None
            The maximum number of bins.
            Number of bins equals number of unique split values n_unique,
            if bins == None or bins > n_unique.
        as_pandas: bool, default True
            Return pd.DataFrame when pandas is installed.
            If False or pandas is not installed, return numpy ndarray.

        Returns
        -------
        a histogram of used splitting values for the specified feature
        either as numpy array or pandas DataFrame.
        """
        ...
    


