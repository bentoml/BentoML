import os
import typing as t

import paddlehub

from bentoml.paddle import PaddleHubModel

test_text: t.List[str] = ["味道不错，确实不算太辣，适合不能吃辣的人。就在长江边上，抬头就能看到长江的风景。鸭肠、黄鳝都比较新鲜。"]


def test_paddlehub_save_load(tmpdir):
    senta = paddlehub.Module(name="senta_bilstm")

    print(senta.sentiment_classify(texts=test_text))
    PaddleHubModel(senta).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "model.pdmodel"))

    paddlehub_loaded = PaddleHubModel.load(tmpdir)
    assert (
        senta.sentiment_classify(texts=test_text)[0]["positive_probs"]
        == paddlehub_loaded.sentiment_classify(texts=test_text)[0]["positive_probs"]
    )
