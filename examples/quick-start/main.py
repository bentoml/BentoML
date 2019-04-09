import os
import sys
from sklearn import svm
from sklearn import datasets

# Use local bentoml code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from bentoml import BentoService, load, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import JsonHandler

@artifacts([PickleArtifact('clf')])
@env(conda_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):
    """
    Iris SVM Classifier
    """

    @api(JsonHandler)
    def predict(self, parsed_json):
        return self.artifacts.clf.predict(parsed_json)

if __name__ == "__main__":
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    model = IrisClassifier.pack(clf=clf)
    print("model.predict = {}".format(model.predict(X[0:1])))

    print("Saving model as bento archive...")
    saved_path = model.save("./model")
    print("BentoML model archive saved to path: {}".format(saved_path))

    bento_model = load(saved_path)
    print("Model output after loading from archiev: {}".format(bento_model.predict(X[0:1])))
