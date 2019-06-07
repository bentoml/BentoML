import os
import sys

from datetime import datetime
import requests
import json
from sklearn import svm
from sklearn import datasets

from clipper_admin import ClipperConnection, DockerContainerManager

# Use local bentoml code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bentoml import BentoService, load, api, env, artifacts, ver
from bentoml.artifact import PickleArtifact
from bentoml.handlers import JsonHandler
from bentoml.deployment.clipper import deploy_bentoml


@artifacts([PickleArtifact("clf")])
@env(conda_dependencies=["scikit-learn"])
@ver(major=1, minor=0)
class IrisClassifier(BentoService):
    """
    Iris SVM Classifier
    """

    @api(JsonHandler)
    def predict(self, parsed_json):
        return self.artifacts.clf.predict(parsed_json)


if __name__ == "__main__":
    clf = svm.SVC(gamma="scale")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    iris_clf_service = IrisClassifier.pack(clf=clf)
    print("iris_clf_service.predict = {}".format(iris_clf_service.predict(X[0:1])))

    print("Saving model as bento archive...")
    saved_path = iris_clf_service.save("/tmp/bento")

    bento_service = load(saved_path)
    print(X[0:1])
    print(
        "Loaded BentoService #predict output: {}".format(bento_service.predict(X[0:1]))
    )

    print("deploy to clipper cluster")
    api_name = "predict"
    clipper_conn = ClipperConnection(DockerContainerManager())
    model_name, model_version = deploy_bentoml(clipper_conn, saved_path, api_name)
    print(
        "Deployed model to clipper. Model: ", model_name, " , version: " + model_version
    )
    print(
        "Register application bento_test_app and link deployed model to the application"
    )
    application_name = "bento_test_app"
    clipper_conn.register_application(
        name=application_name,
        input_type="strings",
        default_output="default result",
        slo_micros=1000000,
    )
    clipper_conn.link_model_to_app(application_name, model_name)
    url = "http://%s/bento_test_app/predict" % clipper_conn.get_query_addr()
    print(
        "Application is registered and successfully linked, you can query it at " + url
    )

    print("Sendint test request to clipper cluster")
    headers = {"Content-type": "application/json"}
    req_json = json.dumps({"input": "test"})
    start = datetime.now()
    request_result = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % (request_result.text, latency))
