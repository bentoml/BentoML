from __future__ import annotations

from typing import TYPE_CHECKING

import bentoml
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.io import Text

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

MODEL_TAG = "blip-image-captioning-large:latest"
PROCESSOR_TAG = "blip-image-captioning-large-processor:latest"


class ImageCaptioningRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = bentoml.transformers.load_model(MODEL_TAG)
        self.processor = bentoml.transformers.load_model(PROCESSOR_TAG)
        self.model.to(self.device)

    @bentoml.Runnable.method(batchable=False)
    def generate(self, img: PILImage, txt: str | None = None) -> str:
        if txt:
            inputs = self.processor(img, txt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(img, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)


runner = bentoml.Runner(
    ImageCaptioningRunnable,
    name="image_captioning_model",
)

svc = bentoml.Service(
    "image_captioning-svc",
    runners=[runner],
)

input_spec = Multipart(img=Image(), prompt=Text())


@svc.api(input=input_spec, output=Text())
async def generate(img: PILImage, prompt: str) -> str:
    return await runner.async_run(img, prompt)
