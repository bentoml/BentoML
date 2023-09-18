from __future__ import annotations

import typing as t

import numpy as np
import torch
import requests
import detectron2.config as Cf
import detectron2.engine as E
import detectron2.modeling as M
import detectron2.model_zoo as Mz
from PIL import Image
from detectron2.data import transforms as T

import bentoml

from . import FrameworkTestModel as Model
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if t.TYPE_CHECKING:
    import torch.nn as nn
    from numpy.typing import NDArray

framework = bentoml.detectron
backward_compatible = False

url = "http://images.cocodataset.org/val2017/000000439715.jpg"

model_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def prepare_input() -> torch.Tensor:
    aug = T.ResizeShortestEdge([800, 800], 1333)
    im: NDArray[np.float32] = np.asarray(
        Image.open(requests.get(url, stream=True).raw).convert("RGB")
    )
    arr: NDArray[np.float32] = aug.get_transform(im).apply_image(im)
    # NOTE: it is fine to copy here
    return torch.as_tensor(arr.transpose(2, 0, 1))


def gen_model() -> tuple[nn.Module, Cf.CfgNode]:
    cfg = Cf.get_cfg()
    cfg.merge_from_file(Mz.get_config_file(model_url))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = Mz.get_checkpoint_url(model_url)
    cfg = cfg.clone()
    cfg.MODEL.DEVICE = "cpu"  # currently only run test on CPU
    m = M.build_model(cfg)
    m.eval()
    return m, cfg


model, cfg = gen_model()

rcnn = Model(
    name="coco-masked-rcnn",
    model=model,
    save_kwargs={"config": cfg},
    configurations=[
        Config(
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[prepare_input()],
                        expected=lambda output: np.testing.assert_allclose(
                            output[0]["instances"].get("scores").tolist()[0], [1.0]
                        ),
                    )
                ]
            }
        )
    ],
)

predictor = Model(
    name="predictor-masked-rcnn",
    model=E.DefaultPredictor(cfg),
    configurations=[
        Config(
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[
                            np.asarray(
                                Image.open(requests.get(url, stream=True).raw).convert(
                                    "RGB"
                                )
                            )
                        ],
                        expected=lambda output: all(
                            map(
                                lambda o: o > 0.5,
                                output["instances"].get("scores").tolist(),
                            )
                        ),
                    )
                ]
            }
        )
    ],
)

models = [rcnn, predictor]
