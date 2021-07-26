import os

import paddlehub as hub
import pytest

from bentoml.paddle import PaddleHubModel

text = ["这部电影太差劲了"]


def get_cut(model, use_gpu=False, batch_size=1, return_tag=False):
    return model.cut(
        text=["今天是个好天气。"], use_gpu=use_gpu, batch_size=batch_size, return_tag=return_tag
    )[0]["word"]


def test_paddlehub_local_dir_save_load(tmpdir):
    senta_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "senta_test")
    PaddleHubModel(senta_path).save(tmpdir)
    hub_loaded = PaddleHubModel.load(tmpdir)

    senta_test = hub.Module(directory=senta_path)
    assert hub_loaded.sentiment_classify(texts=text) == senta_test.sentiment_classify(
        texts=text
    )


@pytest.mark.skip(reason="API is WIP")
def test_paddlehub_online_reg_save_load(tmpdir):
    PaddleHubModel("lac").save(tmpdir)
    hub_loaded = PaddleHubModel.load(tmpdir)
    lac_test = hub.Module(name="lac")
    print(get_cut(lac_test))
    assert all(get_cut(lac_test) == get_cut(hub_loaded))
