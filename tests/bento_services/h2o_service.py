import bentoml
import h2o  # pylint: disable=import-error
from bentoml.adapters import DataframeInput
from bentoml.h2o import H2oModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([H2oModelArtifact('model')])
class H2oExampleBentoService(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        hf = h2o.H2OFrame(df)
        predictions = self.artifacts.model.predict(hf)
        result = predictions.as_data_frame().to_json(orient='records')
        return result
