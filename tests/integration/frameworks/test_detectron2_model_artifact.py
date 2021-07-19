import os
import typing as t

import torch.nn
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

if t.TYPE_CHECKING:
    from detectron2.config import CfgNode  # pylint: disable=unused-import

from bentoml.detectron import DetectronModel


def test_detectron2_save_load(tmpdir):
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

    DetectronModel(
        model, input_model_yaml=model_zoo.get_config_file(model_url), name="mask_rcnn"
    ).save(tmpdir)

    assert os.path.exists(DetectronModel.get_path(tmpdir, ".yaml"))

    detectron_loaded: torch.nn.Module = DetectronModel.load(
        tmpdir, device=cloned.MODEL.DEVICE
    )
    assert repr(detectron_loaded) == repr(model)
