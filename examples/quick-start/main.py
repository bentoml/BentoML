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

    iris_clf_service = IrisClassifier.pack(clf=clf)
    print("iris_clf_service.predict = {}".format(iris_clf_service.predict(X[0:1])))

    print("Saving model as bento archive...")
    saved_path = iris_clf_service.save("/tmp/bento")
    print("BentoML model archive saved to path: {}".format(saved_path))

    bento_service = load(saved_path)
    print("Loaded BentoService #predict output: {}".format(bento_service.predict(X[0:1])))
