import torch

import bentoml
from bentoml.io import Image
from bentoml.io import PandasDataFrame


class Yolov5Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

        # Config inference settings
        self.inference_size = 320

        # Optional configs
        # self.model.conf = 0.25  # NMS confidence threshold
        # self.model.iou = 0.45  # NMS IoU threshold
        # self.model.agnostic = False  # NMS class-agnostic
        # self.model.multi_label = False  # NMS multiple labels per box
        # self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        # self.model.max_det = 1000  # maximum number of detections per image
        # self.model.amp = False  # Automatic Mixed Precision (AMP) inference

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs):
        # Return predictions only
        results = self.model(input_imgs, size=self.inference_size)
        return results.pandas().xyxy

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs):
        # Return images with boxes and labels
        return self.model(input_imgs, size=self.inference_size).render()


yolo_v5_runner = bentoml.Runner(Yolov5Runnable, max_batch_size=30)

svc = bentoml.Service("yolo_v5_demo", runners=[yolo_v5_runner])


@svc.api(input=Image(), output=PandasDataFrame())
async def invocation(input_img):
    batch_ret = await yolo_v5_runner.inference.async_run([input_img])
    return batch_ret[0]


@svc.api(input=Image(), output=Image())
async def render(input_img):
    batch_ret = await yolo_v5_runner.render.async_run([input_img])
    return batch_ret[0]
