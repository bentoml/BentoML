from __future__ import annotations

import numpy as np
import torch
import pandas as pd
import sklearn.datasets as datasets
from fastai.metrics import accuracy
from fastai.torch_core import set_seed
from fastai.tabular.all import tabular_learner
from fastai.tabular.all import TabularDataLoaders

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.fastai

set_seed(123, reproducible=True)

# read in data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
y = pd.Series(iris.target, name="label")
dl = TabularDataLoaders.from_df(
    df=pd.concat([X, y], axis=1), cont_names=list(X.columns), y_names="label"
)
model = tabular_learner(dl, metrics=accuracy, layers=[3])
model.fit(1)


def expected_function(out: tuple[pd.Series, torch.Tensor, torch.Tensor]) -> None:
    res = out[2].numpy()
    assert np.isclose(res, [-0.35807556]).all()


iris_model = FrameworkTestModel(
    name="iris",
    model=model,
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[X.iloc[0]],
                        expected=expected_function,
                    ),
                ],
            },
        ),
    ],
)
models: list[FrameworkTestModel] = [iris_model]
