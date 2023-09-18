from __future__ import annotations

import os
import json
import time
import typing as t
from pathlib import Path

import cv2
import attr
import numpy as np
import torch
import torchvision
from PIL import Image as PILImage

if t.TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch.jit._script import ScriptModule

    P = t.ParamSpec("P")

WORKING_DIR = Path(__file__).parent
MODEL_TYPE = os.environ.get("MODEL_TYPE", "yolov5s")
MODEL_FILE = WORKING_DIR / f"{MODEL_TYPE}.pt"
TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_traced_script() -> tuple[ScriptModule, dict[str, t.Any]]:
    extra_files = {"config.txt": ""}
    model = torch.jit.load(
        MODEL_FILE.with_suffix(".torchscript").__fspath__(),
        map_location=TORCH_DEVICE,
        _extra_files=extra_files,
    )
    d = {"shape": [], "stride": 32, "names": {}}
    if extra_files["config.txt"]:
        d = json.loads(
            extra_files["config.txt"],
            object_hook=lambda d: {
                int(k) if k.isdigit() else k: v for k, v in d.items()
            },
        )
    return model, d


def torch_tensor_from_numpy(
    x: torch.Tensor | NDArray[t.Any], device: str = "cpu"
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        # Needs to create a copy if the array is not writeable
        x = np.copy(x) if not x.flags["WRITEABLE"] else x
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError(
            f"Expected numpy.ndarray or torch.Tensor, got {type(x).__name__}"
        )


def prepare_yolov5_input(
    im_: PILImage.Image,
    size: int | tuple[int, int] = (640, 640),
    auto: bool = False,
    device: str = "cpu",
    fp16: bool = False,
    stride: int = 32,
) -> TensorContainer:
    p = torch.empty(1, device=torch.device(device))
    # Automatic Mixed Precision (AMP) inference
    autocast = fp16 and p.device.type != "cpu"

    im0 = np.asarray(exif_transpose(im_))
    # padded resize
    im, _, _ = letterbox(im0, size, stride=stride, auto=auto)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(p.device)
    im = im.half() if fp16 else im.float()  # unit8 to fp16/fp32
    im /= 255  # 0-255 to 0.0-1.0
    if len(im.shape) == 3:
        im = im[None]  # extends to batch_dim

    return TensorContainer(im, im0, autocast)


def postprocess_yolov5_prediction(
    y: list[t.Any] | tuple[t.Any, ...] | torch.Tensor | NDArray[t.Any],
    prep: TensorContainer,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list[int] | None = None,
    agnostic_nms: bool = False,
    max_det: int = 1000,
):
    res: dict[str, str] = {}
    names = load_traced_script()[1]["names"]
    if isinstance(y, (list, tuple)):
        y = (
            torch_tensor_from_numpy(y[0])
            if len(y) == 1
            else [torch_tensor_from_numpy(_) for _ in y]
        )
    else:
        y = torch_tensor_from_numpy(y)

    y = non_max_suppression(
        y, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
    )
    # post
    for pred in y:
        if len(pred):
            pred[:, :4] = scale_boxes(
                prep.im.shape[2:], pred[:, :4], prep.im0.shape
            ).round()

            klass: int
            for klass in pred[:, 5].unique():
                n: torch.Tensor = (pred[:, 5] == klass).sum()
                res[str(n.item())] = names[int(klass)]
    return res


@attr.define
class TensorContainer:
    im: torch.Tensor
    im0: NDArray[t.Any]
    autocast: bool  # AMP inference

    def to_numpy(self) -> NDArray[t.Any]:
        return self.im.cpu().numpy()


# Some of the function below vendored from yolov5.utils.general


def exif_transpose(image: PILImage.Image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: PILImage.FLIP_LEFT_RIGHT,
            3: PILImage.ROTATE_180,
            4: PILImage.FLIP_TOP_BOTTOM,
            5: PILImage.TRANSPOSE,
            6: PILImage.ROTATE_270,
            7: PILImage.TRANSVERSE,
            8: PILImage.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def letterbox(
    im: t.Any,
    new_shape: tuple[int, int] | int | list[int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[NDArray[t.Any], tuple[float, float], tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x: torch.Tensor | NDArray[t.Any]):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def scale_boxes(
    img1_shape: tuple[int, ...],
    boxes: torch.Tensor,
    img0_shape: tuple[int, ...],
    ratio_pad: torch.Tensor | None = None,
) -> torch.Tensor:
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes: torch.Tensor | NDArray[t.Any], shape: tuple[int, ...]) -> None:
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(
    prediction: t.Any,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list[int] | None = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: tuple[torch.Tensor, ...] = (),
    max_det: int = 300,
    nm: int = 0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
