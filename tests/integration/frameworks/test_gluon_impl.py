import mxnet
import pytest

from bentoml.gluon import GluonModel
from tests.utils.helpers import assert_have_file_extension


@pytest.fixture()
def train_gluon_classifier() -> "mxnet.gluon.nn.HybridSequential":
    net = mxnet.gluon.nn.HybridSequential()
    net.hybridize()
    net.forward(mxnet.nd.array(0))
    return net


def test_gluon_save_pack(tmpdir, train_gluon_classifier):
    GluonModel(train_gluon_classifier).save(tmpdir)
    assert_have_file_extension(tmpdir, "-symbol.json")

    gluon_loaded: "mxnet.gluon.Block" = GluonModel.load(tmpdir)
    assert gluon_loaded(mxnet.nd.array([0])).asnumpy() == [0]
