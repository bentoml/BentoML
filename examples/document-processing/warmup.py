from __future__ import annotations

import requests

import bentoml

ocr_model = "microsoft/trocr-base-handwritten"

if __name__ == "__main__":
    from PIL import Image
    from transformers import TrOCRProcessor
    from transformers import LayoutLMv2Processor
    from transformers import VisionEncoderDecoderModel

    layoutlm_processor = LayoutLMv2Processor.from_pretrained(
        "microsoft/layoutlmv2-base-uncased"
    )
    processor = TrOCRProcessor.from_pretrained(ocr_model)
    model = VisionEncoderDecoderModel.from_pretrained(ocr_model)

    # load image from the IAM dataset
    url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
    generated_text = processor.batch_decode(
        model.generate(
            processor(
                Image.open(requests.get(url, stream=True).raw).convert("RGB"),
                return_tensors="pt",
            ).pixel_values
        ),
        skip_special_tokens=True,
    )[0]

    print(f"\nProcessed text from {url}: {generated_text}\n\n{'-' * 80}\n")
    try:
        bento_model = bentoml.transformers.get("document-processing")
        print(f"Pretrained model '{bento_model.tag.name!s}' already exists.")
    except bentoml.exceptions.NotFound:
        print(
            "Saved pretrained model:",
            bentoml.transformers.import_pretrained(
                "document-processing", processor=processor, model=model
            ),
        )
