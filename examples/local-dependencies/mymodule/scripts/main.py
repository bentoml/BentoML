import os
import sys
import tempfile
from sklearn import svm
from sklearn import datasets

# Use local bentoml code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from bentoml import BentoService, load, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import JsonHandler

# Simulating when user manually add project path to sys.path, and invoke
# script as `python ./mymodule/scripts/main.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mymodule import method_in_mymodule
from mymodule.submodule import method_in_submodule
from mymodule.submodule1 import method_in_submodule1
from mymodule.submodule.submodule2 import method_in_submodule2


@artifacts([PickleArtifact('clf')])
@env(conda_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):
    """
    Iris SVM Classifier
    """

    @api(JsonHandler)
    def predict(self, parsed_json):
        data = method_in_mymodule(parsed_json)
        data = method_in_submodule(data)
        data = method_in_submodule1(data)
        data = method_in_submodule2(data)
        return self.artifacts.clf.predict(data)


if __name__ == "__main__":
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    iris_clf_service = IrisClassifier.pack(clf=clf)

    saved_path = iris_clf_service.save(tempfile.mkdtemp())
    print("Saving new bento service archive to: '{}'".format(saved_path))

    loaded_service = load(saved_path)
    print(loaded_service.predict(X[0:1]))

