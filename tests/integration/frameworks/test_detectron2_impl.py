import typing as t

import imageio
import numpy as np
import pytest
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model

import bentoml.detectron

if t.TYPE_CHECKING:
    from bentoml._internal.models import ModelInfo, ModelStore

if t.TYPE_CHECKING:
    from detectron2.config import CfgNode  # pylint: disable=unused-import

TEST_MODEL_NAME = __name__.split(".")[-1]

IMAGE_URL = "http://images.cocodataset.org/val2017/000000439715.jpg"


def extract_result(raw_result: t.Dict) -> t.Dict:
    pred_instances = raw_result["instances"]
    boxes = pred_instances.pred_boxes.to("cpu").tensor.detach().numpy()
    scores = pred_instances.scores.to("cpu").detach().numpy()
    pred_classes = pred_instances.pred_classes.to("cpu").detach().numpy()

    result = {
        "boxes": boxes,
        "scores": scores,
        "classes": pred_classes,
    }
    return result


def prepare_image(original_image: np.ndarray) -> np.ndarray:
    """Mainly to test on COCO dataset"""
    _aug = T.ResizeShortestEdge([800, 800], 1333)

    image = _aug.get_transform(original_image).apply_image(original_image)
    return image.transpose(2, 0, 1)


def detectron_model_and_config() -> t.Tuple[torch.nn.Module, "CfgNode"]:
    model_url: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    cfg: "CfgNode" = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    # set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)

    cloned = cfg.clone()
    cloned.MODEL.DEVICE = "cpu"  # running on CI
    model: torch.nn.Module = build_model(cloned)
    model.eval()

    return model, cfg


@pytest.fixture(scope="module")
def image_array():
    return np.asarray(imageio.imread(IMAGE_URL))


@pytest.fixture(scope="module")
def save_proc(
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "ModelInfo"]:
    def _(metadata) -> "ModelInfo":
        model, cfg = detectron_model_and_config()
        tag = bentoml.detectron.save(
            TEST_MODEL_NAME,
            model,
            model_config=cfg,
            metadata=metadata,
            model_store=modelstore,
        )
        info = modelstore.get(tag)
        return info

    return _


@pytest.mark.parametrize("metadata", [({"acc": 0.876},)])
def test_detectron2_save_load(metadata, image_array, modelstore, save_proc):

    model, _ = detectron_model_and_config()

    info = save_proc(metadata)
    assert info.metadata is not None

    detectron_loaded = bentoml.detectron.load(
        info.tag,
        device="cpu",
        model_store=modelstore,
    )
    assert next(detectron_loaded.parameters()).device.type == "cpu"
    assert repr(detectron_loaded) == repr(model)

    image = prepare_image(image_array)
    image = torch.as_tensor(image)
    input_data = [{"image": image}]

    raw_result = detectron_loaded(input_data)
    result = extract_result(raw_result[0])
    assert all(i > 0.9 for i in result['scores'])
