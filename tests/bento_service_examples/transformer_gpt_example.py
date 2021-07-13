import bentoml
from bentoml.adapters import JsonInput
from bentoml.transformers import TransformersModelArtifact


@bentoml.env(pip_packages=["torch==1.6.0", "transformers==3.1.0"])
@bentoml.artifacts([TransformersModelArtifact("gptModel")])
class TransformersGPT2TextGenerator(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        src_text = parsed_json.get("text")
        model = self.artifacts.gptModel.get("model")
        tokenizer = self.artifacts.gptModel.get("tokenizer")
        input_ids = tokenizer.encode(src_text, return_tensors="pt")
        output = model.generate(input_ids, max_length=50)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        return output
