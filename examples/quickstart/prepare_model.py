import joblib
from sklearn import datasets
from sklearn import svm

import bentoml

if __name__ == "__main__":
    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target  # type: ignore

    # Model Training
    clf = svm.SVC()
    clf.fit(X, y)

    # Save model to BentoML local model store
    with bentoml.models.create("iris_clf") as bento_model:
        joblib.dump(clf, bento_model.path_of("model.pkl"))
    print(f"Model saved: {bento_model}")
