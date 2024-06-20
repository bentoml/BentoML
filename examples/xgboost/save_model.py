import typing as t

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

import bentoml

# Load the data
cancer: Bunch = t.cast("Bunch", load_breast_cancer())
cancer_data = t.cast("ext.NpNDArray", cancer.data)
cancer_target = t.cast("ext.NpNDArray", cancer.target)
dt = xgb.DMatrix(cancer_data, label=cancer_target)

# Specify model parameters
param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}

# Train the model
model = xgb.train(param, dt)

# Specify the model name and the model to be saved
bentoml.xgboost.save_model("cancer", model)
