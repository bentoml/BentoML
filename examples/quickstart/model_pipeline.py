import bentoml_io as bentoml
import joblib
from sklearn import datasets
from sklearn import svm

from bentoml.models import ModelContext

if __name__ == "__main__":
    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target  # type: ignore

    # Model Training
    clf = svm.SVC()
    clf.fit(X, y)

    # Save model to BentoML local model store
    with bentoml.models.create(
        "iris_clf",
        module="bentoml.pytorch",
        context=ModelContext(framework_name="", framework_versions={}),
        signatures={},
    ) as bento_model:
        joblib.dump(clf, bento_model.path_of("model.pkl"))
    print(f"Model saved: {bento_model}")
