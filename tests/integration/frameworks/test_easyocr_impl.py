import typing as t

import numpy as np
import pytest
import imageio

import bentoml
import easyocr

if t.TYPE_CHECKING:
    from bentoml._internal.models import Model

TEST_MODEL_NAME = __name__.split(".")[-1]
TEST_RESULT: t.List[str] = ["西", "愚园路", "东", "315", "309", "W", "Yuyuan Rd。", "E"]
IMAGE_PATH: str = "./tests/utils/_static/chinese.jpg"

LANG_LIST = ["ch_sim", "en"]
RECOG_NETWORK = "zh_sim_g2"
DETECT_MODEL = "craft_mlt_25k"


def extract_result(raw_result: np.ndarray) -> t.List[str]:
    return [x[1] for x in raw_result]


def easyocr_model() -> easyocr.Reader:
    language_list = LANG_LIST
    recog_network = RECOG_NETWORK

    model = easyocr.Reader(
        lang_list=language_list,
        gpu=False,
        download_enabled=True,
        recog_network=recog_network,
    )

    return model


@pytest.fixture(scope="module")
def image_array():
    return np.asarray(imageio.imread(IMAGE_PATH))


@pytest.fixture(scope="module")
def save_proc() -> t.Callable[
    [
        t.Dict[str, t.Any],
        t.Dict[str, t.Any],
        t.Optional[t.Dict[str, str]],
        t.Optional[t.Dict[str, t.Any]],
    ],
    "Model",
]:
    def _(
        lang_list,
        recog_network,
        detect_model,
        metadata,
        labels=None,
        custom_objects=None,
    ) -> "Model":
        model = easyocr_model()
        tag = bentoml.easyocr.save(
            TEST_MODEL_NAME,
            model,
            lang_list=lang_list,
            recog_network=recog_network,
            detect_model=detect_model,
            metadata=metadata,
            labels=labels,
            custom_objects=custom_objects,
        )
        model = bentoml.models.get(tag)
        return model

    return _


@pytest.mark.parametrize("metadata", [{"acc": 0.876}])
def test_easyocr_save_load(metadata, image_array, save_proc):

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    model = easyocr_model()
    raw_res = model.readtext(IMAGE_PATH)
    assert extract_result(raw_res) == TEST_RESULT

    _model = save_proc(
        LANG_LIST, RECOG_NETWORK, DETECT_MODEL, metadata, labels, {"func": custom_f}
    )
    assert _model.info.metadata is not None
    for k in labels.keys():
        assert labels[k] == _model.info.labels[k]
    assert _model.custom_objects["func"](3) == custom_f(3)

    easyocr_loaded = bentoml.easyocr.load(_model.tag)

    raw_res = easyocr_loaded.readtext(image_array)

    assert extract_result(raw_res) == TEST_RESULT
