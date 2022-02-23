import os

import cv2

import bentoml

text = ["这部电影太差劲了"]

current_dir = os.path.dirname(os.path.realpath(__file__))


def test_paddlehub_local_dir_save_load():
    senta_path = os.path.join(current_dir, "senta_test")
    tag = bentoml.paddle.import_from_paddlehub(senta_path)
    module = bentoml.paddle.load(tag)
    assert module.sentiment_classify(texts=text)[0]["sentiment"] == "negative"


def test_paddlehub_pretrained_save_load():
    tag = bentoml.paddle.import_from_paddlehub("humanseg_lite")
    hub_loaded = bentoml.paddle.load(tag)
    im = cv2.imread(os.path.join(current_dir, "test_image.jpg"))
    res = hub_loaded.segment(images=[im])[0]["data"]
    assert res.shape[0] == 2941
    assert res.shape[1] == 2205
