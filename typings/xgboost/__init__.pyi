import os

from . import dask, rabit, tracker
from .core import Booster, DataIter, DeviceQuantileDMatrix, DMatrix
from .tracker import RabitTracker
from .training import cv, train

"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""
VERSION_FILE = ...
__all__ = [
    "DMatrix",
    "DeviceQuantileDMatrix",
    "Booster",
    "DataIter",
    "train",
    "cv",
    "RabitTracker",
    "XGBModel",
    "XGBClassifier",
    "XGBRegressor",
    "XGBRanker",
    "XGBRFClassifier",
    "XGBRFRegressor",
    "plot_importance",
    "plot_tree",
    "to_graphviz",
    "dask",
    "set_config",
    "get_config",
    "config_context",
]
