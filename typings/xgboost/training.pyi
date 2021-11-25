

"""Training Library containing training routines."""
def train(params, dtrain, num_boost_round=..., evals=..., obj=..., feval=..., maximize=..., early_stopping_rounds=..., evals_result=..., verbose_eval=..., xgb_model=..., callbacks=...):
    """Train a booster with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    evals: list of pairs (DMatrix, string)
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        Requires at least one item in **evals**.
        The method returns the model from the last iteration (not the best one).  Use
        custom callback or model slicing if the best model is desired.
        If there's more than one item in **evals**, the last entry will be used for early
        stopping.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
        If early stopping occurs, the model will have three additional fields:
        ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.  Use
        ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree`` and/or
        ``num_class`` appears in the parameters.  ``best_ntree_limit`` is the result of
        ``num_parallel_tree * best_iteration``.
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist.

        Example: with a watchlist containing
        ``[(dtest,'eval'), (dtrain,'train')]`` and
        a parameter containing ``('eval_metric': 'logloss')``,
        the **evals_result** returns

        .. code-block:: python

            {'train': {'logloss': ['0.48253', '0.35953']},
             'eval': {'logloss': ['0.480385', '0.357756']}}

    verbose_eval : bool or int
        Requires at least one item in **evals**.
        If **verbose_eval** is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If **verbose_eval** is an integer then the evaluation metric on the validation set
        is printed at every given **verbose_eval** boosting stage. The last boosting stage
        / the boosting stage found by using **early_stopping_rounds** is also printed.
        Example: with ``verbose_eval=4`` and at least one item in **evals**, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    xgb_model : file name of stored xgb model or 'Booster' instance
        Xgb model to be loaded before training (allows training continuation).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.
        Example:

        .. code-block:: python

            [xgb.callback.LearningRateScheduler(custom_rates)]

    Returns
    -------
    Booster : a trained booster model
    """
    ...

class CVPack:
    """"Auxiliary datastruct to hold one fold of CV."""
    def __init__(self, dtrain, dtest, param) -> None:
        """"Initialize the CVPack"""
        ...
    
    def __getattr__(self, name): # -> (*args: Unknown, **kwargs: Unknown) -> Any:
        ...
    
    def update(self, iteration, fobj): # -> None:
        """"Update the boosters for one iteration"""
        ...
    
    def eval(self, iteration, feval): # -> str:
        """"Evaluate the CVPack for one iteration."""
        ...
    


class _PackedBooster:
    def __init__(self, cvfolds) -> None:
        ...
    
    def update(self, iteration, obj): # -> None:
        '''Iterate through folds for update'''
        ...
    
    def eval(self, iteration, feval): # -> list[Unknown]:
        '''Iterate through folds for eval'''
        ...
    
    def set_attr(self, **kwargs): # -> None:
        '''Iterate through folds for setting attributes'''
        ...
    
    def attr(self, key):
        '''Redirect to booster attr.'''
        ...
    
    def set_param(self, params, value=...): # -> None:
        """Iterate through folds for set_param"""
        ...
    
    def num_boosted_rounds(self):
        '''Number of boosted rounds.'''
        ...
    
    @property
    def best_iteration(self): # -> int:
        '''Get best_iteration'''
        ...
    
    @property
    def best_score(self): # -> float:
        """Get best_score."""
        ...
    


def groups_to_rows(groups, boundaries):
    """
    Given group row boundaries, convert ground indexes to row indexes
    :param groups: list of groups for testing
    :param boundaries: rows index limits of each group
    :return: row in group
    """
    ...

def mkgroupfold(dall, nfold, param, evals=..., fpreproc=..., shuffle=...): # -> list[Unknown]:
    """
    Make n folds for cross-validation maintaining groups
    :return: cross-validation folds
    """
    ...

def mknfold(dall, nfold, param, seed, evals=..., fpreproc=..., stratified=..., folds=..., shuffle=...):
    """
    Make an n-fold list of CVPack from random indices.
    """
    ...

def cv(params, dtrain, num_boost_round=..., nfold=..., stratified=..., folds=..., metrics=..., obj=..., feval=..., maximize=..., early_stopping_rounds=..., fpreproc=..., as_pandas=..., verbose_eval=..., show_stdv=..., seed=..., callbacks=..., shuffle=...):
    """Cross-validation with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round : int
        Number of boosting iterations.
    nfold : int
        Number of folds in CV.
    stratified : bool
        Perform stratified sampling.
    folds : a KFold or StratifiedKFold instance or list of fold indices
        Sklearn KFolds or StratifiedKFolds object.
        Alternatively may explicitly pass sample indices for each fold.
        For ``n`` folds, **folds** should be a length ``n`` list of tuples.
        Each tuple is ``(in,out)`` where ``in`` is a list of indices to be used
        as the training samples for the ``n`` th fold and ``out`` is a list of
        indices to be used as the testing samples for the ``n`` th fold.
    metrics : string or list of strings
        Evaluation metrics to be watched in CV.
    obj : function
        Custom objective function.
    feval : function
        Custom evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Cross-Validation metric (average of validation
        metric computed over CV folds) needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        The last entry in the evaluation history will represent the best iteration.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
    fpreproc : function
        Preprocessing function that takes (dtrain, dtest, param) and returns
        transformed versions of those.
    as_pandas : bool, default True
        Return pd.DataFrame when pandas is installed.
        If False or pandas is not installed, return np.ndarray
    verbose_eval : bool, int, or None, default None
        Whether to display the progress. If None, progress will be displayed
        when np.ndarray is returned. If True, progress will be displayed at
        boosting stage. If an integer is given, progress will be displayed
        at every given `verbose_eval` boosting stage.
    show_stdv : bool, default True
        Whether to display the standard deviation in progress.
        Results are not affected, and always contains std.
    seed : int
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.
        Example:

        .. code-block:: python

            [xgb.callback.LearningRateScheduler(custom_rates)]
    shuffle : bool
        Shuffle data before creating folds.

    Returns
    -------
    evaluation history : list(string)
    """
    ...

