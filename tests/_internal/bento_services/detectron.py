# TODO: can be used to test service
import typing as t

import numpy as np
import torch
import torch.nn as nn
from detectron2.data import transforms as T


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
