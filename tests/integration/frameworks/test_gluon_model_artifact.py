import os

import mxnet  # pylint: disable=import-error
import pytest

from bentoml.gluon import GluonModel


@pytest.fixture()
def train_gluon_classifier() -> "mxnet.gluon.nn.HybridSequential":
    net = mxnet.gluon.nn.HybridSequential()
    net.hybridize()
    net.forward(mxnet.nd.array(0))
    return net


def test_gluon_save_pack(tmpdir, train_gluon_classifier):
    GluonModel(train_gluon_classifier, name="classifier").save(tmpdir)
    assert os.path.exists(GluonModel.get_path(tmpdir, ".json"))

    gluon_loaded: "mxnet.gluon.Block" = GluonModel.load(tmpdir)
    assert gluon_loaded(mxnet.nd.array([0])).asnumpy() == [0]
