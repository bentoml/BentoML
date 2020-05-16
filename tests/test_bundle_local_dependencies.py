import subprocess

from sklearn import svm
from sklearn import datasets


from tests.conftest import delete_saved_bento_service


def run_test_with_bento_service_class(bento_service_class):
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

    with subprocess.Popen(
        run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        output = proc.stdout.read().decode('utf-8')
        assert output.replace('\r\n', '\n') == '[0]\n'

    delete_saved_bento_service(bento_service.name, bento_service.version)


def test_bundle_local_dependencies():
    from tests.bento_service_examples.local_dependencies.my_test_bento_service import (
        IrisClassifier,
    )

    run_test_with_bento_service_class(IrisClassifier)


def test_bundle_local_dependencies_with_modified_sys_path():
    from tests.bento_service_examples.bento_service_with_modified_sys_path import (
        IrisClassifier,
    )

    run_test_with_bento_service_class(IrisClassifier)
