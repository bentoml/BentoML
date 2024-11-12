from __future__ import annotations

import os

import numpy as np

import bentoml

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
LOG_PATH = os.environ.get("MONITORING_LOG_PATH", "/tmp/iris_monitoring")


@bentoml.service
class IrisClassifier:
    iris_model = bentoml.models.BentoModel("iris_clf:latest")

    def __init__(self):
        self.iris_clf = bentoml.sklearn.load_model(self.iris_model)

    @bentoml.api
    def classify(self, features: np.ndarray) -> str:
        with bentoml.monitor(
            "iris_classifier_prediction", monitor_options={"log_path": LOG_PATH}
        ) as mon:
            mon.log(
                features[0], name="sepal length", role="feature", data_type="numerical"
            )
            mon.log(
                features[1], name="sepal width", role="feature", data_type="numerical"
            )
            mon.log(
                features[2], name="petal length", role="feature", data_type="numerical"
            )
            mon.log(
                features[3], name="petal width", role="feature", data_type="numerical"
            )

            result = self.iris_clf.predict([features])[0]
            category = CLASS_NAMES[result]

            mon.log(category, name="pred", role="prediction", data_type="categorical")
        return category
