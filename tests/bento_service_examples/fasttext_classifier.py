import bentoml

from bentoml.adapters import JsonInput
from bentoml.fasttext import FasttextModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([FasttextModelArtifact('model')])
class FasttextClassifier(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        return self.artifacts.model.predict(parsed_json['text'])
