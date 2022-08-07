from pprint import pprint

import numpy as np
import mlflow
from utils import fetch_logged_data
from sklearn.linear_model import LinearRegression

import bentoml


def main():
    # enable autologging
    mlflow.sklearn.autolog()

    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # train a model
    model = LinearRegression()
    model.fit(X, y)
    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    # import logged MLFlow model to BentoML
    artifact_path = "model"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    bento_model = bentoml.mlflow.import_model("logistic_regression_model", model_uri)
    print("\nModel imported to BentoML: %s" % bento_model)


if __name__ == "__main__":
    main()
