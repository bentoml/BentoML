from __future__ import annotations

import os
import uuid
import typing as t
import platform
import subprocess

import dit
import numpy as np
import torch
import easyocr
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import bentoml

if t.TYPE_CHECKING:
    from detectron2.config import CfgNode
    from detectron2.engine import DefaultPredictor
    from detectron2.structures import Boxes
    from detectron2.structures import Instances


def convert_pdf_to_images(
    pdf_path: str | bytes, **convert_attrs: t.Any
) -> list[Image.Image]:
    try:
        subprocess.check_output(["pdfinfo", "-v"], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "Make sure to install 'poppler' on macOS with brew: 'brew install poppler'"
            )
        elif platform.system() == "Windows":
            raise RuntimeError(
                "Refer to https://github.com/Belval/pdf2image for Windows instruction."
            )
        else:
            raise RuntimeError(
                "'pdftocairo' and 'pdftoppm' should already be included in your Linux distrobution (Seems like they are not installed). Refer to your package manager and install 'poppler-utils'"
            )
    try:
        import pdf2image
    except ImportError:
        raise RuntimeError(
            "Make sure to install all required dependencies with 'pip install -r requirements.txt'."
        )
    if not isinstance(pdf_path, (str, bytes)):
        raise TypeError(
            "pdf_path should be either a path to a PDF file or a bytes object containing a PDF file."
        )
    convert_attrs.setdefault("thread_count", 6)

    fn = (
        pdf2image.convert_from_bytes
        if isinstance(pdf_path, bytes)
        else pdf2image.convert_from_path
    )
    return fn(pdf_path, **convert_attrs)


def play_segmentation(
    im: Image.Image, predictor: DefaultPredictor, cfg: CfgNode, visualize: bool = False
) -> tuple[list[float], list[float], Boxes]:
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    tensor = np.array(im)
    output: Instances = predictor(tensor)["instances"]
    if visualize:
        v = Visualizer(
            tensor[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
        )
        res = v.draw_instance_predictions(output.to("cpu"))
        Image.fromarray(res.get_image()[:, :, ::-1]).save(
            f"{uuid.uuid4()}-segmented.png"
        )
    return (
        output.get("pred_classes").tolist(),
        output.get("scores").tolist(),
        output.get("pred_boxes"),
    )


@torch.inference_mode()
def main(threshold: float = 0.8, analyze: bool = False):
    # TODO: support EOL token.
    reader = easyocr.Reader(["en"])
    cfg = dit.get_cfg()
    predictor = dit.get_predictor(cfg)

    if analyze:
        print(
            "\nUsing EasyOCR model with LayouLMv3 Detectron2 model for PDF extraction.",
        )
        print("=" * 50)
        res = []
        for i, im in enumerate(
            convert_pdf_to_images(os.path.join("samples", "2204.08387.pdf"), dpi=300)
        ):  # layoutlmv3 paper
            print(f"Processing page {i}...")
            classes, scores, boxes = play_segmentation(im, predictor, cfg)
            for cls, score, box in zip(classes, scores, boxes):
                # We don't care about table in this case and any prediction lower than the given threshold
                if cls != 4 and score >= threshold:
                    cropped = im.crop(box.numpy())
                    # join text if it's in the same line
                    join_char = "" if cls == 0 else " "
                    text = join_char.join(
                        [t[1] for t in reader.readtext(np.asarray(cropped))]
                    )
                    # ignore annotations for table footer
                    if not text.startswith("Figure"):
                        print("Extract text:", text)
                        res.append(text)
            print(f"Done processing page {i}.")
            print("-" * 25, "\n")
        print("Finished processing all pages.")
        print("=" * 50)

    try:
        reader_model = bentoml.easyocr.get("en-reader")
        print(f"'en-reader' is previously saved: {reader_model}")
    except bentoml.exceptions.NotFound:
        reader_model = bentoml.easyocr.save_model("en-reader", reader)
        print(f"'en-reader' is saved: {reader_model}")

    dit_tag = "dit-predictor"
    try:
        predictor_model = bentoml.detectron.get(dit_tag)
        print(f"'{dit_tag}' is previously saved: {predictor_model}")
    except bentoml.exceptions.NotFound:
        predictor_model = bentoml.detectron.save_model(dit_tag, predictor)
        print(f"'{dit_tag}' is saved: {predictor_model}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    raise SystemExit(main(**vars(args)))
