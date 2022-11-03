import numpy as np

import bentoml
from bentoml.io import Text
from bentoml.io import NumpyNdarray

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])


@svc.api(
    input=NumpyNdarray.from_sample(np.array([4.9, 3.0, 1.4, 0.2], dtype=np.double)),
    output=Text(),
)
async def classify(input_series: np.ndarray) -> np.ndarray:
    with bentoml.monitor("iris_classifier_prediction") as mon:
        mon.log(
            input_series[0],
            name="sepal length",
            role="feature",
            data_type="numerical",
        )
        mon.log(
            input_series[1],
            name="sepal width",
            role="feature",
            data_type="numerical",
        )
        mon.log(
            input_series[2],
            name="petal length",
            role="feature",
            data_type="numerical",
        )
        mon.log(
            input_series[3],
            name="petal width",
            role="feature",
            data_type="numerical",
        )
        result = await iris_clf_runner.predict.async_run([input_series])[0]
        category = CLASS_NAMES[result]
        mon.log(
            category,
            name="pred",
            role="prediction",
            data_type="categorical",
        )
    return category
