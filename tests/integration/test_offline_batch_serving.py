import logging
import os
import subprocess
import pytest
import imageio
import csv
import numpy as np

from sklearn import datasets, svm

from tests.conftest import delete_saved_bento_service
from tests.bento_service_examples import (
    pytorch_classifier_image,
    iris_classifier,
)

logger = logging.getLogger('bentoml.test')
test_data = "[[5.1, 3.5, 1.4, 0.2]]"


@pytest.fixture()
def csv_file(tmpdir):
    csv_file_ = tmpdir.join("test_csv.csv")
    with open(csv_file_, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, 1, 2, 3])
        writer.writerow([5.1, 3.5, 1.4, 0.2])
    return str(csv_file_)


@pytest.fixture()
def img_file(tmpdir):
    img_file_ = tmpdir.join("test_img.jpg")
    imageio.imwrite(str(img_file_), np.zeros((32, 32, 3)))
    return str(img_file_)


def assert_out(stdout, stderr, expected):
    assert not stderr
    assert stdout.strip().split('\n')[-1] == expected


# generic run predict
def run_predict(bento_service, input_data, is_file=False, options=[]):
    bento_name = f"{bento_service.name}:{bento_service.version}"
    run_predict_deployment_command = [
        'bentoml',
        'run',
        bento_name,
        'predict',
        *options,
        '--input-file' if is_file else '--input',
        input_data,
    ]

    logger.info(f'Run predict command: {run_predict_deployment_command}')
    try:
        with subprocess.Popen(
            run_predict_deployment_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        ) as proc:
            stdout = proc.stdout.read().decode('utf-8')
            stderr = proc.stderr.read().decode('utf-8')
        logger.info(f'Got output: {stdout}')
        return stdout, stderr
    finally:
        delete_saved_bento_service(bento_service.name, bento_service.version)


def test_run_predict_input_dataframe_adapter():
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    bento_service = iris_classifier.IrisClassifier()
    bento_service.pack('model', clf)
    bento_service.save()

    stdout, stderr = run_predict(bento_service, test_data)
    assert_out(stdout, stderr, '[0]')


def test_run_predict_input_file_adapter(csv_file):
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    bento_service = iris_classifier.IrisClassifier()
    bento_service.pack('model', clf)
    bento_service.save()

    stdout, stderr = run_predict(
        bento_service, csv_file, is_file=True, options=['--format', 'csv']
    )
    assert_out(stdout, stderr, '[0]')


def test_run_predict_input_image_adapter(img_file, trained_pytorch_classifier):
    bento_service = pytorch_classifier_image.PytorchImageClassifier()
    bento_service.pack('net', trained_pytorch_classifier)
    bento_service.save()

    stdout, stderr = run_predict(bento_service, img_file, is_file=True)
    assert_out(stdout, stderr, '"deer"')
