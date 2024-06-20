from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

import bentoml

iris = load_iris()
X_train = iris.data[:, :4]
Y_train = iris.target

model_uri = Path("models", "IrisClf")
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
mlflow.sklearn.save_model(model, model_uri.resolve())
# model_uri can be any URI that refers to an MLflow model
# Use local path for demostration
bentoml.mlflow.import_model("iris", model_uri)
