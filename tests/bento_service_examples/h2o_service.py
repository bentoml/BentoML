import h2o

import bentoml
from bentoml.artifact import H2oModelArtifact
from bentoml.adapters import DataframeInput


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([H2oModelArtifact('model')])
class H2oExampleBentoService(bentoml.BentoService):
    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        hf = h2o.H2OFrame(df)
        predictions = self.artifacts.model.predict(hf)
        result = predictions.as_data_frame().to_json(orient='records')
        return result
