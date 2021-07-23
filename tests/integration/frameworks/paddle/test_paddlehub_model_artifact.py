import os

import paddlehub as hub

from bentoml.paddle import PaddleHubModel

text = ["这部电影太差劲了"]


def test_paddlehub_save_load(tmpdir):
    senta_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "senta_test")
    PaddleHubModel(senta_path).save(tmpdir)
    hub_loaded = PaddleHubModel.load(tmpdir)

    senta_test = hub.Module(directory=senta_path)
    assert hub_loaded.sentiment_classify(texts=text) == senta_test.sentiment_classify(
        texts=text
    )
