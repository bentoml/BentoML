import subprocess

from sklearn import svm
from sklearn import datasets

from tests.bento_service_examples.local_dependencies.my_test_bento_service import (
    IrisClassifier,
)


def test_bundle_local_dependencies():
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    # Create a iris classifier service
    iris_classifier_service = IrisClassifier()

    # Pack it with the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to a BentoService bundle
    iris_classifier_service.save()

    bento_name = f"{iris_classifier_service.name}:{iris_classifier_service.version}"

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
        assert output == '[0]\n'
