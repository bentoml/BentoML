import pytest
import time
import urllib

import xgboost as xgb
import pandas as pd

import bentoml
from tests.bento_service_examples.xgboost_classifier import XgboostModelClassifier


def XgboostModel():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # read in data
    cancer = load_breast_cancer()

    X = cancer.data
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    # specify parameters via map
    param = {'max_depth': 3, 'eta': 0.3, 'objective': 'multi:softprob', 'num_class': 2}
    num_round = 20
    bst = xgb.train(param, dtrain, num_round)

    return bst


test_data = {
    "mean radius": 10.80,
    "mean texture": 21.98,
    "mean perimeter": 68.79,
    "mean area": 359.9,
    "mean smoothness": 0.08801,
    "mean compactness": 0.05743,
    "mean concavity": 0.03614,
    "mean concave points": 0.2016,
    "mean symmetry": 0.05977,
    "mean fractal dimension": 0.3077,
    "radius error": 1.621,
    "texture error": 2.240,
    "perimeter error": 20.20,
    "area error": 20.02,
    "smoothness error": 0.006543,
    "compactness error": 0.02148,
    "concavity error": 0.02991,
    "concave points error": 0.01045,
    "symmetry error": 0.01844,
    "fractal dimension error": 0.002690,
    "worst radius": 12.76,
    "worst texture": 32.04,
    "worst perimeter": 83.69,
    "worst area": 489.5,
    "worst smoothness": 0.1303,
    "worst compactness": 0.1696,
    "worst concavity": 0.1927,
    "worst concave points": 0.07485,
    "worst symmetry": 0.2965,
    "worst fractal dimension": 0.07662,
}


@pytest.fixture(scope="module")
def xgboost_svc():
    svc = XgboostModelClassifier()
    model = XgboostModel()
    svc.pack('model', model)

    return svc


@pytest.fixture(scope="module")
def xgboost_svc_saved_dir(tmp_path_factory, xgboost_svc):
    tmpdir = str(tmp_path_factory.mktemp("xgboost_svc"))
    xgboost_svc.save_to_dir(tmpdir)
    return tmpdir


@pytest.fixture()
def xgboost_svc_loaded(xgboost_svc_saved_dir):
    return bentoml.load(xgboost_svc_saved_dir)


def test_xgboost_artifact(xgboost_svc_loaded):
    test_df = pd.DataFrame([test_data])
    result = xgboost_svc_loaded.predict(test_df)
    assert result == 1


@pytest.fixture()
def xgboost_image(xgboost_svc_saved_dir):
    import docker

    client = docker.from_env()

    image = client.images.build(
        path=xgboost_svc_saved_dir, tag='xgboost_example_service', rm=True
    )[0]
    yield image
    client.images.remove(image.id)


def _wait_until_ready(_host, timeout, check_interval=0.5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if (
                urllib.request.urlopen(f'http://{_host}/healthz', timeout=0.1).status
                == 200
            ):
                break
        except Exception:
            time.sleep(check_interval - 0.1)
        else:
            raise AssertionError(f"server didn't get ready in {timeout} seconds")


@pytest.fixture()
def xgboost_docker_host(xgboost_image):
    import docker

    client = docker.from_env()
    with bentoml.utils.reserve_free_port() as port:
        pass

    command = 'bentoml serve-gunicorn /bento --workers 1'

    try:
        container = client.containers.run(
            command=command,
            image=xgboost_image.id,
            auto_remove=True,
            tty=True,
            ports={'5000/tcp': port},
            detach=True,
        )
        _host = f'127.0.0.1:{port}'
        _wait_until_ready(_host, 10)
        yield _host
    finally:
        container.stop()
        time.sleep(1)


# @pytest.mark.asyncio
def test_api_server_with_docker(xgboost_docker_host):
    import requests

    test_df = pd.DataFrame([test_data])

    response = requests.post(
        f"http://{xgboost_docker_host}/predict",
        headers={"Content-Type": "application/json"},
        data=test_df.to_json(),
    )

    preds = response.json()
    assert response.status_code == 200
    assert preds[0] == 1
