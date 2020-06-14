import bentoml
from bentoml.adapters import (
    DataframeInput,
    ImageInput,
    LegacyImageInput,
    JsonInput,
    # FastaiImageInput,
)
from bentoml.handlers import DataframeHandler  # deprecated
from bentoml.artifact import PickleArtifact


@bentoml.artifacts([PickleArtifact("model")])
@bentoml.env(auto_pip_dependencies=True)
class ExampleBentoService(bentoml.BentoService):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        """An API for testing simple bento model service
        """
        return self.artifacts.model.predict(df)

    @bentoml.api(input=DataframeInput(input_dtypes={"col1": "int"}))
    def predict_dataframe(self, df):
        """predict_dataframe expects dataframe as input
        """
        return self.artifacts.model.predict_dataframe(df)

    @bentoml.api(DataframeHandler, input_dtypes={"col1": "int"})  # deprecated
    def predict_dataframe_v1(self, df):
        """predict_dataframe expects dataframe as input
        """
        return self.artifacts.model.predict_dataframe(df)

    @bentoml.api(input=ImageInput())
    def predict_image(self, images):
        return self.artifacts.model.predict_image(images)

    @bentoml.api(input=LegacyImageInput(input_names=('original', 'compared')))
    def predict_images(self, original, compared):
        return original[0, 0] == compared[0, 0]

    @bentoml.api(input=JsonInput())
    def predict_json(self, input_data):
        return self.artifacts.model.predict_json(input_data)

    # Disabling fastai related tests to fix travis build
    # @bentoml.api(input=FastaiImageInput())
    # def predict_fastai_image(self, input_data):
    #     return self.artifacts.model.predict_image(input_data)
    #
    # @bentoml.api(input=FastaiImageInput(input_names=('original', 'compared')))
    # def predict_fastai_images(self, original, compared):
    #     return all(original.data[0, 0] == compared.data[0, 0])
