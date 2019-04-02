import os
import sys
from sklearn import svm
from sklearn import datasets

from bentoml import BentoModel, handler, load
from bentoml.artifacts import PickleArtifact
from bentoml.handlers import JsonHandler

class IrisClassifier(BentoModel):
    """
    Iris SVM Classifier
    """
    
    def config(self, artifacts, env):
        artifacts.add(PickleArtifact('clf'))
        env.add_conda_dependencies(["scikit-learn"])

    @handler(JsonHandler)
    def predict(self, parsed_json):
        return self.artifacts.clf.predict(parsed_json)

if __name__ == "__main__":
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    model = IrisClassifier(clf=clf)
    print("model.predict = {}".format(model.predict(X[0:1])))

    saved_path = model.save("./model")
    print("BentoML model archive saved to path: {}".format(saved_path))

    bento_model = load(saved_path)
    print("Model output after loading from archiev: {}".format(bento_model.predict(X[0:1])))
