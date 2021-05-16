import bentoml
from bentoml.adapters import ImageInput
from bentoml.frameworks.easyocr import EasyOCRArtifact

import numpy as np


@bentoml.env(pip_packages=["easyocr>=1.3.0"])
@bentoml.artifacts([EasyOCRArtifact("chinese_small")])
class EasyOCRService(bentoml.BentoService):
    @bentoml.api(input=ImageInput(), batch=False)
    def predict(self, image):
        reader = self.artifacts.chinese_small
        raw_results = reader.readtext(np.array(image))
        text_instances = [x[1] for x in raw_results]
        return {"text": text_instances}
