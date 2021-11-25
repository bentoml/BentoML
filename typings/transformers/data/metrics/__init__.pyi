

import warnings

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

from ...file_utils import is_sklearn_available, requires_backends

if is_sklearn_available():
    ...
DEPRECATION_WARNING = ...
def simple_accuracy(preds, labels):
    ...

def acc_and_f1(preds, labels):
    ...

def pearson_and_spearman(preds, labels):
    ...

def glue_compute_metrics(task_name, preds, labels):
    ...

def xnli_compute_metrics(task_name, preds, labels):
    ...

