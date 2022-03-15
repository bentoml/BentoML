import typing as t
from typing import TYPE_CHECKING

import numpy as np
import torch
import pytest
import imageio
from detectron2 import model_zoo
from detectron2.data import transforms as T
from detectron2.config import get_cfg
from detectron2.modeling import build_model

import bentoml

if TYPE_CHECKING:
    from detectron2.config import CfgNode

    from bentoml import Tag


IMAGE_URL: str = "./tests/utils/_static/detectron2_sample.jpg"


def extract_result(raw_result: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
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


def prepare_image(
    original_image: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
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


@pytest.fixture(scope="module", name="image_array")
def fixture_image_array() -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    return np.asarray(imageio.imread(IMAGE_URL))


def save_procedure(
    metadata: t.Dict[str, t.Any],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
) -> "Tag":
    model, config = detectron_model_and_config()
    tag_info = bentoml.detectron.save(
        "test_detectron2_model",
        model,
        model_config=config,
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    )
    return tag_info


@pytest.mark.parametrize("metadata", [{"acc": 0.876}])
def test_detectron2_save_load(
    metadata: t.Dict[str, t.Any],
    image_array: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> None:

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    tag = save_procedure(
        metadata,
        labels=labels,
        custom_objects={"func": custom_f},
    )
    _model = bentoml.models.get(tag)

    assert _model.info.metadata is not None

    detectron_loaded = bentoml.detectron.load(
        _model.tag,
        device="cpu",
    )
    assert next(detectron_loaded.parameters()).device.type == "cpu"

    image = prepare_image(image_array)
    image = torch.as_tensor(image)
    input_data = [{"image": image}]

    raw_result = detectron_loaded(input_data)
    result = extract_result(raw_result[0])
    assert result is not None

    for k in labels.keys():
        assert labels[k] == _model.info.labels[k]
    assert _model.custom_objects["func"](3) == custom_f(3)


def test_detectron2_setup_run_batch(
    image_array: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> None:
    tag = save_procedure({})
    runner = bentoml.detectron.load_runner(tag)
    assert tag in runner.required_models
    assert runner.num_replica == 1
    image = torch.as_tensor(prepare_image(image_array))
    res = runner.run_batch(image)
    result = extract_result(res[0])
    assert result["boxes"] is not None
