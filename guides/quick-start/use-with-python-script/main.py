from sklearn import svm
from sklearn import datasets

from bentoml import BentoService, load, api, env, artifacts, ver
from bentoml.artifact import PickleArtifact
from bentoml.handlers import JsonHandler


@artifacts([PickleArtifact('clf')])
@env(pip_dependencies=["scikit-learn"])
@ver(major=1, minor=0)
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

    print("Saving BentoService file bundle...")
    saved_path = iris_clf_service.save()
    print("BenteService bundle created at path: {}".format(saved_path))

    loaded_bento_service = load(saved_path)
    print(X[0:1])
    print("Loaded BentoService #predict output: {}".format(loaded_bento_service.predict(X[0:1])))
