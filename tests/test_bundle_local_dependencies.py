import subprocess
import pytest
import logging

from sklearn import datasets, svm

from tests.conftest import delete_saved_bento_service
from tests.bento_services.local_dependencies import (
    bento_service_with_zipimport,
    my_test_bento_service,
)
from tests.bento_services import bento_service_with_modified_sys_path


service_classes = [
    pytest.param(bento_service_with_zipimport.IrisClassifier, id='zipimports'),
    pytest.param(my_test_bento_service.IrisClassifier, id='local modules'),
    pytest.param(
        bento_service_with_modified_sys_path.IrisClassifier, id='modified sys path'
    ),
]


@pytest.mark.parametrize("bento_service_class", service_classes)
def test_bento_service_class(bento_service_class):
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    # Create a bento service instance
    bento_service = bento_service_class()

    # Pack it with the newly trained model artifact
    bento_service.pack('model', clf)

    # Save the prediction service to a BentoService bundle
    bento_service.save()

    bento_name = f"{bento_service.name}:{bento_service.version}"

    run_command = [
        "bentoml",
        "run",
        bento_name,
        "predict",
        "--input",
        "[[5.1, 3.5, 1.4, 0.2]]",
        "-q",
    ]
    print(f"running command {' '.join(run_command)}:")

    try:
        with subprocess.Popen(
            run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as proc:
            output = proc.stdout.read().decode('utf-8')
            err_msg = proc.stderr.read().decode('utf-8')
            logging.warning(err_msg)
            assert not err_msg
            assert output.strip() == '[0]'
    finally:
        delete_saved_bento_service(bento_service.name, bento_service.version)
