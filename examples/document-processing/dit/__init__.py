"""
All sources under this folder are excerpted from https://github.com/microsoft/unilm/tree/master/dit.
This is to avoid cloning the whole repo to use this model only.
"""
from __future__ import annotations

import os

import torch
from detectron2.config import CfgNode
from detectron2.config import get_cfg as _get_cfg
from detectron2.engine import DefaultPredictor

from .backbone import build_vit_fpn_backbone as build_vit_fpn_backbone


def get_cfg(cfg: CfgNode | None = None) -> CfgNode:
    if cfg is None:
        cfg = _get_cfg()

    add_vit_config(cfg)
    cfg.merge_from_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascade_dit_base.yml")
    )

    # Step 2: add model weights URL to config
    cfg.MODEL.WEIGHTS = (
        "https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth"
    )

    # Step 3: set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def get_predictor(cfg: CfgNode | None = None) -> DefaultPredictor:
    return DefaultPredictor(get_cfg(cfg))


def add_vit_config(cfg: CfgNode):
    """Add config for VIT. From https://github.com/microsoft/unilm to avoid cloning the whole repo for this function only."""
    _C = cfg

    _C.MODEL.VIT = CfgNode()

    # CoaT model name.
    _C.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]

    _C.MODEL.VIT.IMG_SIZE = [224, 224]

    _C.MODEL.VIT.POS_TYPE = "shared_rel"

    _C.MODEL.VIT.DROP_PATH = 0.0

    _C.MODEL.VIT.MODEL_KWARGS = "{}"

    _C.SOLVER.OPTIMIZER = "ADAMW"

    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0

    _C.AUG = CfgNode()

    _C.AUG.DETR = False


__all__ = ["build_vit_fpn_backbone", "get_predictor"]
