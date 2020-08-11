import os

from sklearn import svm
from sklearn import datasets

from bentoml.saved_bundle import load_bento_service_metadata
from tests.bento_service_examples.iris_classifier import IrisClassifier
from tests.conftest import delete_saved_bento_service


def test_auto_adapter_dependencies(bento_bundle_path):
    with open(os.path.join(bento_bundle_path, 'requirements.txt')) as f:
        requirements_txt_content = f.read()

    dependencies = requirements_txt_content.split('\n')
    dependencies = [dep.split('==')[0] for dep in dependencies]
    assert 'imageio' in dependencies
    assert 'bentoml' in dependencies

    # Test that dependencies also wrote to BentoServiceMetadat config file


def test_auto_artifact_dependencies():
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

    with open(os.path.join(saved_path, 'requirements.txt')) as f:
        requirements_txt_content = f.read()

    dependencies = requirements_txt_content.split('\n')
    dependencies = [dep.split('==')[0] for dep in dependencies]
    assert 'scikit-learn' in dependencies
    assert 'bentoml' in dependencies

    # Test that dependencies also wrote to BentoServiceMetadat config file
    bs_matadata = load_bento_service_metadata(saved_path)
    dependencies = bs_matadata.env.pip_dependencies.split('\n')
    dependencies = [dep.split('==')[0] for dep in dependencies]
    assert 'scikit-learn' in dependencies
    assert 'bentoml' in dependencies

    delete_saved_bento_service(
        iris_classifier_service.name, iris_classifier_service.version
    )
