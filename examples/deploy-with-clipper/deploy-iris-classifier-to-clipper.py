import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn import datasets
from datetime import datetime
import requests
import json

from clipper_admin import ClipperConnection, DockerContainerManager

from bentoml import load
from bentoml.deployment.clipper import deploy_bentoml


if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    saved_path = "./model"

    bento_service = load(saved_path)
    print(X[0:1])
    print(
        "Loaded BentoService #predict output: {}".format(bento_service.predict(X[0:1]))
    )

    # Deploying to local clipper cluster with BentoML archive.
    print("deploy to clipper cluster")
    api_name = "predict"
    clipper_conn = ClipperConnection(DockerContainerManager())

    print("Stop all existing clipper deployments")
    clipper_conn.stop_all()
    print("Deploying model to clipper")
    model_name, model_version = deploy_bentoml(
        clipper_conn,
        saved_path,
        api_name,
        'floats'
    )
    print(
        "Deployed model to clipper. Model: ", model_name, " , version: " + model_version
    )
    print(
        "Register application bento_test_app and link deployed model to the application"
    )
    application_name = "bentoml_test_app"
    clipper_conn.register_application(
        name=application_name,
        input_type="floats",
        default_output="default result",
        slo_micros=10000,
    )
    clipper_conn.link_model_to_app(application_name, model_name)
    url = "http://%s/bento_test_app/predict" % clipper_conn.get_query_addr()
    print(
        "Application is registered and successfully linked, you can query it at " + url
    )

    print("Sending test request to clipper cluster")
    #   headers = {"Content-type": "application/json"}
    #   req_json = json.dumps({"input": X[0:1].tolist()})
    #   start = datetime.now()
    #   request_result = requests.post(url, headers=headers, data=req_json)
    #   end = datetime.now()
    #   latency = (end - start).total_seconds() * 1000.0
    print("RESULT from clipper: '%s', LATENCY: %f ms" % (request_result.text, latency))
