from __future__ import annotations

import sys
from typing import Any

import numpy as np
import PIL.Image

import bentoml
from bentoml.io import JSON
from bentoml.io import Image

yolo_runner = bentoml.pytorch.get("pytorch_yolov5").to_runner()

svc = bentoml.Service(name="pytorch_yolo_demo", runners=[yolo_runner])


sys.path.append("yolov5")


@svc.api(input=Image(), output=JSON())
async def predict_image(img: PIL.Image.Image) -> list[Any]:
    assert isinstance(img, PIL.Image.Image)
    return await yolo_runner.async_run([np.array(img)])
