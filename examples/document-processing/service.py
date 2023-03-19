from __future__ import annotations

from PIL import Image

import bentoml


class DocAIRunnable(
    bentoml.transformers.PreTrainedRunnable,
    models=["document-processing:toji36ggmo6vfgxi"],
):
    @bentoml.Runnable.method(batchable=False)
    def image_to_text(self, input_img: Image.Image) -> list[str]:
        return self.document_processing.processor.batch_decode(
            self.document_processing.model.generate(
                self.document_processing.processor(
                    input_img.convert("RGB"), return_tensors="pt"
                ).pixel_values
            ),
            skip_special_tokens=True,
        )


docai_runner = bentoml.Runner(DocAIRunnable)

svc = bentoml.Service(name="document-processing", runners=[docai_runner])


@svc.api(input=bentoml.io.Image(), output=bentoml.io.Text())
async def predict(img: Image.Image) -> str:
    res = await docai_runner.image_to_text.async_run(img)
    return res[0]
