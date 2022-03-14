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
def test_gluon_save_load(train_gluon_classifier, metadata):

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    model = train_gluon_classifier

    tag = bentoml.gluon.save(
        TEST_MODEL_NAME,
        model,
        metadata=metadata,
        labels=labels,
        custom_objects={"func": custom_f},
    )

    gluon_loaded: gluon.Block = bentoml.gluon.load(tag)
    assert gluon_loaded(mxnet.nd.array([0])).asnumpy() == [0]

    bentomodel = bentoml.models.get(tag)
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)
