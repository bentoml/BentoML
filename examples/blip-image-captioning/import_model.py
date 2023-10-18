from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration

import bentoml

model_id = "Salesforce/blip-image-captioning-large"

processor = AutoProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)


bentoml.transformers.save_model("blip-image-captioning-large", model)
bentoml.transformers.save_model("blip-image-captioning-large-processor", processor)
