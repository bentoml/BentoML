import bentoml
from bentoml.adapters import JsonInput
from bentoml.spacy import SpacyModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([SpacyModelArtifact('nlp')])
class SpacyModelService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        output = self.artifacts.nlp(parsed_json['text'])
        return output
