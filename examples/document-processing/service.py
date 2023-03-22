from __future__ import annotations

import io
import os
import typing as t

import numpy as np
from warmup import convert_pdf_to_images

import bentoml

THRESHOLD = os.getenv("OCR_THRESHOLD", 0.8)

en_reader = bentoml.easyocr.get("en-reader").to_runner()
processor = bentoml.detectron.get("dit-predictor").to_runner()


svc = bentoml.Service(name="document-processing", runners=[en_reader, processor])


@svc.api(input=bentoml.io.File(), output=bentoml.io.JSON())
async def image_to_text(file: io.BytesIO) -> dict[t.Literal["parsed"], str]:
    res = []
    with file:
        ims = convert_pdf_to_images(file.read())
        for im in ims:
            output = (await processor.async_run(np.asarray(im)))["instances"]
            segmentation = (
                output.get("pred_classes").tolist(),
                output.get("scores").tolist(),
                output.get("pred_boxes"),
            )
            for cls, score, box in zip(*segmentation):
                # We don't care about table in this case and any prediction lower than the given threshold
                if cls != 4 and score >= THRESHOLD:
                    # join text if it's in the same line
                    text = " ".join(
                        [
                            t[1]
                            for t in await en_reader.readtext.async_run(
                                np.asarray(im.crop(box.numpy()))
                            )
                        ]
                    )
                    # ignore annotations for table footer
                    if not text.startswith("Figure"):
                        print("Extract text:", text)
                        res.append(text)
        return {"parsed": "\n".join(res)}
