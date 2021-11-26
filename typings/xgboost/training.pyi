def train(
    params,
    dtrain,
    num_boost_round=...,
    evals=...,
    obj=...,
    feval=...,
    maximize=...,
    early_stopping_rounds=...,
    evals_result=...,
    verbose_eval=...,
    xgb_model=...,
    callbacks=...,
): ...

class CVPack:
    def __init__(self, dtrain, dtest, param) -> None: ...
    def __getattr__(self, name): ...
    def update(self, iteration, fobj): ...
    def eval(self, iteration, feval): ...

class _PackedBooster:
    def __init__(self, cvfolds) -> None: ...
    def update(self, iteration, obj): ...
    def eval(self, iteration, feval): ...
    def set_attr(self, **kwargs): ...
    def attr(self, key): ...
    def set_param(self, params, value=...): ...
    def num_boosted_rounds(self): ...
    @property
    def best_iteration(self): ...
    @property
    def best_score(self): ...

def groups_to_rows(groups, boundaries): ...
def mkgroupfold(dall, nfold, param, evals=..., fpreproc=..., shuffle=...): ...
def mknfold(
    dall,
    nfold,
    param,
    seed,
    evals=...,
    fpreproc=...,
    stratified=...,
    folds=...,
    shuffle=...,
): ...
def cv(
    params,
    dtrain,
    num_boost_round=...,
    nfold=...,
    stratified=...,
    folds=...,
    metrics=...,
    obj=...,
    feval=...,
    maximize=...,
    early_stopping_rounds=...,
    fpreproc=...,
    as_pandas=...,
    verbose_eval=...,
    show_stdv=...,
    seed=...,
    callbacks=...,
    shuffle=...,
): ...
