from pprint import pprint

import mlflow
import pandas as pd
from utils import fetch_logged_data
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

import bentoml


def main():
    mlflow.sklearn.autolog()

    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    clf.fit(iris.data, iris.target)
    run_id = mlflow.last_active_run().info.run_id

    # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    # show data logged in the child runs
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in parameters.keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs[["run_id", *param_cols, *metric_cols]])

    # import only the best_estimator artifact to BentoML
    artifact_path = "best_estimator"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    bento_model = bentoml.mlflow.import_model("sklearn_gs_iris", model_uri)
    print("\nModel imported to BentoML: %s" % bento_model)


if __name__ == "__main__":
    main()
