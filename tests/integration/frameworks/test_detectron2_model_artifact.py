import os
import typing as t

import numpy as np
import pytest
import torch
import torch.nn
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model

from bentoml.detectron import DetectronModel

if t.TYPE_CHECKING:
    from detectron2.config import CfgNode  # pylint: disable=unused-import


@pytest.fixture(scope='session')
def predict_image(
    model: nn.Module, original_image: np.ndarray
) -> t.Dict[str, np.ndarray]:
    """Mainly to test on COCO dataset"""
    _aug = T.ResizeShortestEdge([800, 800], 1333)

    height, width = original_image.shape[:2]
    image = _aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    inputs = {"image": image, "height": height, "width": width}
    pred = model([inputs])[0]
    pred_instances = pred["instances"]
    boxes = pred_instances.pred_boxes.to("cpu").tensor.detach().numpy()
    scores = pred_instances.scores.to("cpu").detach().numpy()
    pred_classes = pred_instances.pred_classes.to("cpu").detach().numpy()

    result = {
        "boxes": boxes,
        "scores": scores,
        "classes": pred_classes,
    }
    return result


def test_detectron2_save_load(tmpdir, predict_image):
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

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    DetectronModel(model, input_model_yaml=model_zoo.get_config_file(model_url)).save(
        tmpdir
    )

    assert os.path.exists(DetectronModel.model_path(tmpdir, ".yaml"))

    detectron_loaded: torch.nn.Module = DetectronModel.load(
        tmpdir, device=cloned.MODEL.DEVICE
    )
    assert repr(detectron_loaded) == repr(model)
