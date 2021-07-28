#  Copyright (c) 2021 Atalaya Tech, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==========================================================================
#

import os

from pkg_resources import parse_requirements
from sklearn import datasets, svm

from bentoml.saved_bundle import load_bento_service_metadata
from tests._internal.bento_services import IrisClassifier, IrisClassifierPipEnv
from tests.unit.conftest import delete_saved_bento_service


def test_auto_adapter_dependencies(bento_bundle_path):
    with open(os.path.join(bento_bundle_path, "requirements.txt")) as f:
        requirements_txt_content = f.read()

    dependencies = requirements_txt_content.split("\n")
    dependencies = [dep.split("==")[0] for dep in dependencies]
    assert "imageio" in dependencies
    assert "bentoml" in dependencies

    # Test that dependencies also wrote to BentoServiceMetadata config file


def _fit_clf():
    clf = svm.SVC(gamma="scale")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)
    return clf


def _assert_in_dependencies(expected, dependencies):
    for dep in expected:
        assert dep in dependencies


def _parse_dependencies(path):
    with open(os.path.join(path, "requirements.txt")) as f:
        requirements_txt_content = f.read()

    return requirements_txt_content.split("\n")


def _dependencies_to_requirements(deps):
    deps = [dep.split("==")[0] for dep in deps if not dep.startswith("-")]
    return [r.name for r in parse_requirements(deps)]


def test_auto_artifact_dependencies():
    clf = _fit_clf()

    # Create a iris classifier service
    iris_classifier_service = IrisClassifier()

    # Pack it with the newly trained model artifact
    iris_classifier_service.pack("model", clf)

    # Save the prediction service to a BentoService bundle
    saved_path = iris_classifier_service.save()

    # parse generated requirements.txt
    dependencies = _dependencies_to_requirements(_parse_dependencies(saved_path))
    _assert_in_dependencies(["scikit-learn", "bentoml"], dependencies)

    # Test that dependencies also wrote to BentoServiceMetadata config file
    bs_metadata = load_bento_service_metadata(saved_path)
    dependencies = bs_metadata.env.pip_packages
    dependencies = _dependencies_to_requirements(dependencies)
    _assert_in_dependencies(["scikit-learn", "bentoml"], dependencies)

    # Clean up
    delete_saved_bento_service(
        iris_classifier_service.name, iris_classifier_service.version
    )


# similar to test_auto_artifact_dependencies
def test_requirements_txt_file():
    clf = _fit_clf()
    iris_classifier_service = IrisClassifierPipEnv()
    iris_classifier_service.pack("model", clf)
    saved_path = iris_classifier_service.save()

    dependencies = _dependencies_to_requirements(_parse_dependencies(saved_path))
    _assert_in_dependencies(
        ["scikit-learn", "azure-cli", "psycopg2-binary", "bentoml"], dependencies
    )

    bs_metadata = load_bento_service_metadata(saved_path)
    requirements_txt = bs_metadata.env.requirements_txt
    requirements_content = open("./tests/pipenv_requirements.txt", "r").read()
    assert requirements_txt == requirements_content

    delete_saved_bento_service(
        iris_classifier_service.name, iris_classifier_service.version
    )
