

from typing import Optional, Tuple, Union

import numpy as np

from .compat import XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .core import Booster, _deprecate_positional_args

"""Scikit-Learn Wrapper interface for XGBoost."""
class XGBRankerMixIn:
    """MixIn for ranking, defines the _estimator_type usually defined in scikit-learn base
    classes."""
    _estimator_type = ...


__estimator_doc = ...
__model_doc = ...
__custom_obj_note = ...
def xgboost_model_doc(header, items, extra_parameters=..., end_note=...): # -> (cls: Unknown) -> Unknown:
    '''Obtain documentation for Scikit-Learn wrappers

    Parameters
    ----------
    header: str
       An introducion to the class.
    items : list
       A list of commom doc items.  Available items are:
         - estimators: the meaning of n_estimators
         - model: All the other parameters
         - objective: note for customized objective
    extra_parameters: str
       Document for class specific parameters, placed at the head.
    end_note: str
       Extra notes put to the end.
'''
    ...

@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost.""", ['estimators', 'model', 'objective'])
class XGBModel(XGBModelBase):
    def __init__(self, max_depth=..., learning_rate=..., n_estimators=..., verbosity=..., objective=..., booster=..., tree_method=..., n_jobs=..., gamma=..., min_child_weight=..., max_delta_step=..., subsample=..., colsample_bytree=..., colsample_bylevel=..., colsample_bynode=..., reg_alpha=..., reg_lambda=..., scale_pos_weight=..., base_score=..., random_state=..., missing=..., num_parallel_tree=..., monotone_constraints=..., interaction_constraints=..., importance_type=..., gpu_id=..., validate_parameters=..., **kwargs) -> None:
        ...
    
    def get_booster(self): # -> Booster:
        """Get the underlying xgboost Booster of this model.

        This will raise an exception when fit was not called

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        ...
    
    def set_params(self, **params): # -> Self@XGBModel:
        """Set the parameters of this estimator.  Modification of the sklearn method to
        allow unknown kwargs. This allows using the full range of xgboost
        parameters that are not defined as member variables in sklearn grid
        search.

        Returns
        -------
        self

        """
        ...
    
    def get_params(self, deep=...):
        """Get parameters."""
        ...
    
    def get_xgb_params(self): # -> dict[Unknown, Unknown]:
        """Get xgboost specific parameters."""
        ...
    
    def get_num_boosting_rounds(self):
        """Gets the number of xgboost boosting rounds."""
        ...
    
    def save_model(self, fname: str): # -> None:
        """Save the model to a file.

        The model is saved in an XGBoost internal format which is universal
        among the various XGBoost interfaces. Auxiliary attributes of the
        Python Booster object (such as feature names) will not be saved.

          .. note::

            See:

            https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

        Parameters
        ----------
        fname : string
            Output file name

        """
        ...
    
    def load_model(self, fname): # -> None:
        """Load the model from a file.

        The model is loaded from an XGBoost internal format which is universal
        among the various XGBoost interfaces. Auxiliary attributes of the
        Python Booster object (such as feature names) will not be loaded.

        Parameters
        ----------
        fname : string
            Input file name.

        """
        ...
    
    @_deprecate_positional_args
    def fit(self, X, y, *, sample_weight=..., base_margin=..., eval_set=..., eval_metric=..., early_stopping_rounds=..., verbose=..., xgb_model: Optional[Union[Booster, str, XGBModel]] = ..., sample_weight_eval_set=..., base_margin_eval_set=..., feature_weights=..., callbacks=...): # -> Self@XGBModel:
        """Fit gradient boosting model.

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            instance weights
        base_margin : array_like
            global bias for each instance.
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        eval_metric : str, list of str, or callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.rst.
            If a list of str, should be the list of multiple built-in evaluation metrics
            to use.
            If callable, a custom evaluation metric. The call signature is
            ``func(y_predicted, y_true)`` where ``y_true`` will be a DMatrix object such
            that you may need to call the ``get_label`` method. It must return a str,
            value pair where the str is a name for the evaluation and value is the value
            of the evaluation function. The callable custom objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            Requires at least one item in **eval_set**.
            The method returns the model from the last iteration (not the best one).
            If there's more than one item in **eval_set**, the last entry will be used
            for early stopping.
            If there's more than one metric in **eval_metric**, the last metric will be
            used for early stopping.
            If early stopping occurs, the model will have three additional fields:
            ``clf.best_score``, ``clf.best_iteration`` and ``clf.best_ntree_limit``.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model :
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is an array like
            object storing instance weights for the i-th validation set.
        base_margin_eval_set : list, optional
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights: array_like
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.  Only available for `hist`, `gpu_hist` and
            `exact` tree methods.
        callbacks : list of callback functions
            List of callback functions that are applied at end of each iteration.
            It is possible to use predefined callbacks by using :ref:`callback_api`.
            Example:

            .. code-block:: python

                callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                        save_best=True)]

        """
        ...
    
    def predict(self, X, output_margin=..., ntree_limit=..., validate_features=..., base_margin=..., iteration_range=...):
        """
        Predict with `X`.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X : array_like
            Data to predict with
        output_margin : bool
            Whether to output the raw untransformed margin value.
        ntree_limit : int
            Deprecated, use `iteration_range` instead.
        validate_features : bool
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin : array_like
            Margin added to prediction.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

            .. versionadded:: 1.4.0
        Returns
        -------
        prediction : numpy array
        """
        ...
    
    def apply(self, X, ntree_limit: int = ..., iteration_range: Optional[Tuple[int, int]] = ...) -> np.ndarray:
        """Return the predicted leaf every tree for each sample.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).

        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
            For each datapoint x in X and for each tree, return the index of the
            leaf x ends up in. Leaves are numbered within
            ``[0; 2**(self.max_depth+1))``, possibly with gaps in the numbering.
        """
        ...
    
    def evals_result(self): # -> dict[Unknown, Unknown]:
        """Return the evaluation results.

        If **eval_set** is passed to the `fit` function, you can call
        ``evals_result()`` to get evaluation results for all passed **eval_sets**.
        When **eval_metric** is also passed to the `fit` function, the
        **evals_result** will contain the **eval_metrics** passed to the `fit` function.

        Returns
        -------
        evals_result : dictionary

        Example
        -------

        .. code-block:: python

            param_dist = {'objective':'binary:logistic', 'n_estimators':2}

            clf = xgb.XGBModel(**param_dist)

            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric='logloss',
                    verbose=True)

            evals_result = clf.evals_result()

        The variable **evals_result** will contain:

        .. code-block:: python

            {'validation_0': {'logloss': ['0.604835', '0.531479']},
            'validation_1': {'logloss': ['0.41965', '0.17686']}}
        """
        ...
    
    @property
    def n_features_in_(self) -> int:
        ...
    
    @property
    def best_score(self) -> float:
        ...
    
    @property
    def best_iteration(self) -> int:
        ...
    
    @property
    def best_ntree_limit(self) -> int:
        ...
    
    @property
    def feature_importances_(self): # -> ndarray[Unknown, Unknown] | Any:
        """
        Feature importances property

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        Returns
        -------
        feature_importances_ : array of shape ``[n_features]``

        """
        ...
    
    @property
    def coef_(self): # -> ndarray[Any, Unknown] | ndarray[Unknown, Unknown]:
        """
        Coefficients property

        .. note:: Coefficients are defined only for linear learners

            Coefficients are only defined when the linear model is chosen as
            base learner (`booster=gblinear`). It is not defined for other base
            learner types, such as tree learners (`booster=gbtree`).

        Returns
        -------
        coef_ : array of shape ``[n_features]`` or ``[n_classes, n_features]``
        """
        ...
    
    @property
    def intercept_(self): # -> ndarray[Unknown, Unknown]:
        """
        Intercept (bias) property

        .. note:: Intercept is defined only for linear learners

            Intercept (bias) is only defined when the linear model is chosen as base
            learner (`booster=gblinear`). It is not defined for other base learner types, such
            as tree learners (`booster=gbtree`).

        Returns
        -------
        intercept_ : array of shape ``(1,)`` or ``[n_classes]``
        """
        ...
    


PredtT = ...
@xgboost_model_doc("Implementation of the scikit-learn API for XGBoost classification.", ['model', 'objective'], extra_parameters='''
    n_estimators : int
        Number of boosting rounds.
    use_label_encoder : bool
        (Deprecated) Use the label encoder from scikit-learn to encode the labels. For new
        code, we recommend that you set this parameter to False.
''')
class XGBClassifier(XGBModel, XGBClassifierBase):
    @_deprecate_positional_args
    def __init__(self, *, objective=..., use_label_encoder=..., **kwargs) -> None:
        ...
    
    @_deprecate_positional_args
    def fit(self, X, y, *, sample_weight=..., base_margin=..., eval_set=..., eval_metric=..., early_stopping_rounds=..., verbose=..., xgb_model=..., sample_weight_eval_set=..., base_margin_eval_set=..., feature_weights=..., callbacks=...):
        ...
    
    def predict(self, X, output_margin=..., ntree_limit=..., validate_features=..., base_margin=..., iteration_range: Optional[Tuple[int, int]] = ...): # -> ndarray[Unknown, Unknown] | Any:
        ...
    
    def predict_proba(self, X, ntree_limit=..., validate_features=..., base_margin=..., iteration_range: Optional[Tuple[int, int]] = ...) -> np.ndarray:
        """ Predict the probability of each `X` example being of a given class.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X : array_like
            Feature matrix.
        ntree_limit : int
            Deprecated, use `iteration_range` instead.
        validate_features : bool
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin : array_like
            Margin added to prediction.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

        Returns
        -------
        prediction : numpy array
            a numpy array of shape array-like of shape (n_samples, n_classes) with the
            probability of each data example being of a given class.
        """
        ...
    
    def evals_result(self): # -> dict[Unknown, Unknown]:
        """Return the evaluation results.

        If **eval_set** is passed to the `fit` function, you can call
        ``evals_result()`` to get evaluation results for all passed **eval_sets**.
        When **eval_metric** is also passed to the `fit` function, the
        **evals_result** will contain the **eval_metrics** passed to the `fit` function.

        Returns
        -------
        evals_result : dictionary

        Example
        -------

        .. code-block:: python

            param_dist = {'objective':'binary:logistic', 'n_estimators':2}

            clf = xgb.XGBClassifier(**param_dist)

            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric='logloss',
                    verbose=True)

            evals_result = clf.evals_result()

        The variable **evals_result** will contain

        .. code-block:: python

            {'validation_0': {'logloss': ['0.604835', '0.531479']},
            'validation_1': {'logloss': ['0.41965', '0.17686']}}
        """
        ...
    


@xgboost_model_doc("scikit-learn API for XGBoost random forest classification.", ['model', 'objective'], extra_parameters='''
    n_estimators : int
        Number of trees in random forest to fit.
    use_label_encoder : bool
        (Deprecated) Use the label encoder from scikit-learn to encode the labels. For new
        code, we recommend that you set this parameter to False.
''')
class XGBRFClassifier(XGBClassifier):
    @_deprecate_positional_args
    def __init__(self, *, learning_rate=..., subsample=..., colsample_bynode=..., reg_lambda=..., use_label_encoder=..., **kwargs) -> None:
        ...
    
    def get_xgb_params(self): # -> dict[Unknown, Unknown]:
        ...
    
    def get_num_boosting_rounds(self): # -> Literal[1]:
        ...
    


@xgboost_model_doc("Implementation of the scikit-learn API for XGBoost regression.", ['estimators', 'model', 'objective'])
class XGBRegressor(XGBModel, XGBRegressorBase):
    @_deprecate_positional_args
    def __init__(self, *, objective=..., **kwargs) -> None:
        ...
    


@xgboost_model_doc("scikit-learn API for XGBoost random forest regression.", ['model', 'objective'], extra_parameters='''
    n_estimators : int
        Number of trees in random forest to fit.
''')
class XGBRFRegressor(XGBRegressor):
    @_deprecate_positional_args
    def __init__(self, *, learning_rate=..., subsample=..., colsample_bynode=..., reg_lambda=..., **kwargs) -> None:
        ...
    
    def get_xgb_params(self): # -> dict[Unknown, Unknown]:
        ...
    
    def get_num_boosting_rounds(self): # -> Literal[1]:
        ...
    


@xgboost_model_doc('Implementation of the Scikit-Learn API for XGBoost Ranking.', ['estimators', 'model'], end_note='''
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
''')
class XGBRanker(XGBModel, XGBRankerMixIn):
    @_deprecate_positional_args
    def __init__(self, *, objective=..., **kwargs) -> None:
        ...
    
    @_deprecate_positional_args
    def fit(self, X, y, *, group=..., qid=..., sample_weight=..., base_margin=..., eval_set=..., eval_group=..., eval_qid=..., eval_metric=..., early_stopping_rounds=..., verbose=..., xgb_model: Optional[Union[Booster, str, XGBModel]] = ..., sample_weight_eval_set=..., base_margin_eval_set=..., feature_weights=..., callbacks=...) -> XGBRanker:
        """Fit gradient boosting ranker

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        group : array_like
            Size of each query group of training data. Should have as many elements as the
            query groups in the training data.  If this is set to None, then user must
            provide qid.
        qid : array_like
            Query ID for each training sample.  Should have the size of n_samples.  If
            this is set to None, then user must provide group.
        sample_weight : array_like
            Query group weights

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group/id (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign weights
                to individual data points.
        base_margin : array_like
            Global bias for each instance.
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        eval_group : list of arrays, optional
            A list in which ``eval_group[i]`` is the list containing the sizes of all
            query groups in the ``i``-th pair in **eval_set**.
        eval_qid : list of array_like, optional
            A list in which ``eval_qid[i]`` is the array containing query ID of ``i``-th
            pair in **eval_set**.
        eval_metric : str, list of str, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.rst.
            If a list of str, should be the list of multiple built-in evaluation metrics
            to use. The custom evaluation metric is not yet supported for the ranker.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.  Requires at
            least one item in **eval_set**.
            The method returns the model from the last iteration (not the best one).  If
            there's more than one item in **eval_set**, the last entry will be used for
            early stopping.
            If there's more than one metric in **eval_metric**, the last metric will be
            used for early stopping.
            If early stopping occurs, the model will have three additional fields:
            ``clf.best_score``, ``clf.best_iteration`` and ``clf.best_ntree_limit``.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model :
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list of
            group weights on the i-th validation set.

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign
                weights to individual data points.
        base_margin_eval_set : list, optional
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights: array_like
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.  Only available for `hist`, `gpu_hist` and
            `exact` tree methods.
        callbacks : list of callback functions
            List of callback functions that are applied at end of each
            iteration.  It is possible to use predefined callbacks by using
            :ref:`callback_api`.  Example:

            .. code-block:: python

                callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                        save_best=True)]

        """
        ...
    


