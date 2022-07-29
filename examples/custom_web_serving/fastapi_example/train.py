import logging

import pandas as pd
from sklearn import svm
from sklearn import datasets

import bentoml

logging.basicConfig(level=logging.WARN)

if __name__ == "__main__":

    # Load training data
    iris = datasets.load_iris()
    X = pd.DataFrame(
        data=iris.data, columns=["sepal_len", "sepal_width", "petal_len", "petal_width"]
    )
    y = iris.target

    # Model Training
    clf = svm.SVC()
    clf.fit(X, y)

    # Save model to BentoML local model store
    saved_model = bentoml.sklearn.save_model("iris_clf_with_feature_names", clf)
    print(f"Model saved: {saved_model}")
