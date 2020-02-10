#!/usr/bin/env python
import subprocess

from sklearn import svm
from sklearn import datasets

import bentoml
from bentoml.handlers import DataframeHandler
from bentoml.artifact import SklearnModelArtifact

from my_test_dependency import dummy_util_func


@bentoml.env(pip_dependencies=["scikit-learn"])
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):
    @bentoml.api(DataframeHandler)
    def predict(self, df):
        df = dummy_util_func(df)

        from dynamically_imported_dependency import dummy_util_func_ii

        df = dummy_util_func_ii(df)

        return self.artifacts.model.predict(df)


if __name__ == "__main__":
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    # Create a iris classifier service
    iris_classifier_service = IrisClassifier()

    # Pack it with the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to a BentoService bundle
    saved_path = iris_classifier_service.save()

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

    from scripts.e2e_tests.cli_operations import delete_bento

    delete_bento(bento_name)
