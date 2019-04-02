import os
import sys
from sklearn import svm
from sklearn import datasets

# use local bentoml repo if not installed
sys.path.append(os.path.join(os.getcwd(), "..", "..")) 
from bentoml import BentoModel, handler, load
from bentoml.artifacts import PickleArtifact
from bentoml.handlers import JsonHandler

from mymodule import method_a
from mymodule2 import method_b
from mymodule.submodule3 import method_c
from mymodule.submodule4 import method_d

class IrisClassifier(BentoModel):
    """
    Iris SVM Classifier
    """
    
    def config(self, artifacts, env):
        artifacts.add(PickleArtifact('clf'))
        env.add_conda_dependencies(["scikit-learn"])

    @handler(JsonHandler)
    def predict(self, parsed_json):
        data = method_a(parsed_json)
        data = method_b(data)
        data = method_c(data)
        data = method_d(data)
        return self.artifacts.clf.predict(data)

if __name__ == "__main__":
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    model = IrisClassifier(clf=clf)
    print(model.__class__.__module__)

    saved_path = model.save("./model")
    print(saved_path)

    # from importlib import import_module
    # m = import_module(model.__class__.__module__)

    # print(m)
    # print(m.__name__)
    # print(m.__file__)
