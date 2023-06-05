from __future__ import annotations

import os

import numpy as np

import bentoml
from bentoml.io import Text
from bentoml.io import NumpyNdarray

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

LOG_PATH = os.environ.get("MONITORING_LOG_PATH", "/tmp/iris_monitoring")


@svc.api(
    input=NumpyNdarray.from_sample(np.array([4.9, 3.0, 1.4, 0.2], dtype=np.double)),
    output=Text(),
)
async def classify(features: np.ndarray) -> str:
    with bentoml.monitor(
        "iris_classifier_prediction", monitor_options={"log_path": LOG_PATH}
    ) as mon:
        mon.log(features[0], name="sepal length", role="feature", data_type="numerical")
        mon.log(features[1], name="sepal width", role="feature", data_type="numerical")
        mon.log(features[2], name="petal length", role="feature", data_type="numerical")
        mon.log(features[3], name="petal width", role="feature", data_type="numerical")

        results = await iris_clf_runner.predict.async_run([features])
        result = results[0]
        category = CLASS_NAMES[result]

        mon.log(category, name="pred", role="prediction", data_type="categorical")
    return category
