import bentoml
from bentoml import PickleArtifact
from bentoml._internal.adapters import (
    DataframeInput,
    ImageInput,
    JsonInput,
    MultiImageInput,
)


@bentoml.artifacts([PickleArtifact("model")])
@bentoml.env(infer_pip_packages=True)
class MockBentoService(bentoml.Service):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(
        input=DataframeInput(), mb_max_latency=1000, mb_max_batch_size=2000, batch=True
    )
    def predict(self, df):
        """An API for testing simple bento model service
        """
        return self.artifacts.model.predict(df)

    @bentoml.api(input=DataframeInput(dtype={"col1": "int"}), batch=True)
    def predict_dataframe(self, df):
        """predict_dataframe expects dataframe as input
        """
        return self.artifacts.model.predict_dataframe(df)

    @bentoml.api(input=ImageInput(), batch=True)
    def predict_image(self, images):
        return self.artifacts.model.predict_image(images)

    @bentoml.api(
        input=MultiImageInput(input_names=('original', 'compared')), batch=False
    )
    def predict_multi_images(self, original, compared):
        return self.artifacts.model.predict_multi_images(original, compared)

    @bentoml.api(input=JsonInput(), batch=True)
    def predict_json(self, input_data):
        return self.artifacts.model.predict_json(input_data)

    CUSTOM_ROUTE = "$~!@%^&*()_-+=[]\\|;:,./predict"

    @bentoml.api(
        route=CUSTOM_ROUTE, input=JsonInput(), batch=True,
    )
    def customize_route(self, input_data):
        return input_data
