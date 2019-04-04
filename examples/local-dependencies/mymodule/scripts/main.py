import os
import sys
from sklearn import svm
from sklearn import datasets

# Use local bentoml code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from bentoml import BentoModel, handler, load
from bentoml.artifacts import PickleArtifact
from bentoml.handlers import JsonHandler

from mymodule import method_in_mymodule
from mymodule.submodule import method_in_submodule
from mymodule.submodule1 import method_in_submodule1
from mymodule.submodule.submodule2 import method_in_submodule2

class IrisClassifier(BentoModel):
    """
    Iris SVM Classifier
    """
    
    def config(self, artifacts, env):
        artifacts.add(PickleArtifact('clf'))
        env.add_conda_dependencies(["scikit-learn"])

    @handler(JsonHandler)
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

    model = IrisClassifier(clf=clf)
    print(model.__class__.__module__)

    saved_path = model.save("./model")
    print("Saving new bento model archive to: '{}'".format(saved_path))

    loaded_model = bentoml.load(saved_path)
    loaded_model.predict()

