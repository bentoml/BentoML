import mxnet
import pytest
from mxnet import gluon

import bentoml

TEST_MODEL_NAME = __name__.split(".")[-1]


@pytest.fixture()
def train_gluon_classifier() -> gluon.nn.HybridSequential:
    net = mxnet.gluon.nn.HybridSequential()
    net.hybridize()
    net.forward(mxnet.nd.array(0))
    return net


@pytest.mark.parametrize("metadata", [{"acc": 0.876}])
def test_gluon_save_load(train_gluon_classifier, metadata, modelstore):

    model = train_gluon_classifier

    tag = bentoml.gluon.save(
        TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
    )

    gluon_loaded: gluon.Block = bentoml.gluon.load(tag, model_store=modelstore)
    assert gluon_loaded(mxnet.nd.array([0])).asnumpy() == [0]
