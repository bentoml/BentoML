# The example is based on the coco example in
# https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/

import torch  # pylint: disable=import-error
import numpy as np
import bentoml
import sys
import traceback
from typing import Dict
from bentoml.detectron import DetectronModelArtifact
from bentoml.adapters import ImageInput
from detectron2.data import transforms as T  # pylint: disable=import-error


def get_traceback_list():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    return traceback.format_exception(exc_type, exc_value, exc_traceback)


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([DetectronModelArtifact("model")])
class DetectronClassifier(bentoml.BentoService):
    @bentoml.api(input=ImageInput(), batch=False)
    def predict(self, original_image: np.ndarray) -> Dict:
        _aug = T.ResizeShortestEdge([800, 800], 1333)

        height, width = original_image.shape[:2]
        image = _aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.artifacts.model([inputs])[0]
        pred_instances = predictions["instances"]
        boxes = (pred_instances.pred_boxes).to("cpu").tensor.detach().numpy()
        scores = (pred_instances.scores).to("cpu").detach().numpy()
        pred_classes = (pred_instances.pred_classes).to("cpu").detach().numpy()
        pred_masks = (pred_instances.pred_masks).to("cpu").detach().numpy()

        result = {
            "boxes": boxes,
            "scores": scores,
            "classes": pred_classes,
            "masks": pred_masks,
        }
        return result
