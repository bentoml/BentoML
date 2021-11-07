import typing as t

import easyocr
import imageio
import numpy as np
import pytest

import bentoml.easyocr

if t.TYPE_CHECKING:
    from bentoml._internal.models.store import ModelInfo, ModelStore

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
def save_proc(
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "ModelInfo"]:
    def _(lang_list, recog_network, detect_model, metadata) -> "ModelInfo":
        model = easyocr_model()
        tag = bentoml.easyocr.save(
            TEST_MODEL_NAME,
            model,
            lang_list=lang_list,
            recog_network=recog_network,
            detect_model=detect_model,
            metadata=metadata,
            model_store=modelstore,
        )
        info = modelstore.get(tag)
        return info

    return _


@pytest.mark.parametrize("metadata", [({"acc": 0.876},)])
def test_easyocr_save_load(metadata, image_array, modelstore, save_proc):

    model = easyocr_model()
    raw_res = model.readtext(IMAGE_PATH)
    assert extract_result(raw_res) == TEST_RESULT

    info = save_proc(LANG_LIST, RECOG_NETWORK, DETECT_MODEL, metadata)
    assert info.metadata is not None

    easyocr_loaded = bentoml.easyocr.load(
        info.tag,
        gpu=False,
        model_store=modelstore,
    )

    raw_res = easyocr_loaded.readtext(image_array)

    assert extract_result(raw_res) == TEST_RESULT
