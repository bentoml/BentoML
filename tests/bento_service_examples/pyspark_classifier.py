import bentoml
from bentoml.handlers import DataframeHandler
from bentoml.artifact import PysparkModelArtifact


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([PysparkModelArtifact('model', spark_version="2.4.0")])
class PysparkClassifier(bentoml.BentoService):
    @bentoml.api(DataframeHandler)
    def predict(self, df):
        model_input = df.to_numpy()
        return self.artifacts.pyspark_model.predict(model_input)
