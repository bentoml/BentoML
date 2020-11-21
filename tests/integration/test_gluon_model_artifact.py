import pytest
import bentoml
from tests.bento_service_examples.gluon_classifier import GluonClassifier
from bentoml.yatai.client import YataiClient

import mxnet


@pytest.fixture()
def gluon_classifier():
    GluonClassifier._bento_service_bundle_path = None
    GluonClassifier._bento_service_bundle_version = None
    return GluonClassifier()


@pytest.fixture()
def trained_gluon_model():
    net = mxnet.gluon.nn.HybridSequential()
    net.hybridize()
    net.forward(mxnet.nd.array(0))
    return net


def test_gluon_artifact_pack(gluon_classifier, trained_gluon_model):
    gluon_classifier.pack('model', trained_gluon_model)

    assert gluon_classifier.predict([0]) == [0]

    saved_path = gluon_classifier.save()
    loaded_svc = bentoml.load(saved_path)

    assert loaded_svc.predict([0]) == [0]

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f'{gluon_classifier.name}:{gluon_classifier.version}')
