from __future__ import annotations

from PIL import Image

import bentoml

trocr_processor = bentoml.transformers.get("trocr-processor").to_runner()
layoutlm_processor = bentoml.transformers.get("layoutlm-processor").to_runner()
ocr_model = bentoml.transformers.get("ocr-model").to_runner()

svc = bentoml.Service(
    name="document-processing", runners=[trocr_processor, layoutlm_processor, ocr_model]
)


@svc.api(input=bentoml.io.Image(), output=bentoml.io.Text())
async def image_to_text(img: Image.Image) -> str:
    pixel_values = (
        await trocr_processor.async_run(img.convert("RGB"), return_tensors="pt")
    ).pixel_values
    res = await trocr_processor.batch_decode.async_run(
        await ocr_model.generate.async_run(pixel_values), skip_special_tokens=True
    )
    return "".join(res)
