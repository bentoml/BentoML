import bentoml
from bentoml.handlers import (
    DataframeHandler,
    ImageHandler,
    JsonHandler,
    # FastaiImageHandler,
)
from bentoml.artifact import PickleArtifact


@bentoml.artifacts([PickleArtifact("model")])
@bentoml.env()
class ExampleBentoService(bentoml.BentoService):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(DataframeHandler)
    def predict(self, df):
        """An API for testing simple bento model service
        """
        return self.artifacts.model.predict(df)

    @bentoml.api(DataframeHandler, input_dtypes={"col1": "int"})
    def predict_dataframe(self, df):
        """predict_dataframe expects dataframe as input
        """
        return self.artifacts.model.predict_dataframe(df)

    @bentoml.api(ImageHandler)
    def predict_image(self, image):
        return self.artifacts.model.predict_image(image)

    @bentoml.api(ImageHandler, input_names=('original', 'compared'))
    def predict_images(self, original, compared):
        return original[0, 0] == compared[0, 0]

    @bentoml.api(JsonHandler)
    def predict_json(self, input_data):
        return self.artifacts.model.predict_json(input_data)

    # Disabling fastai related tests to fix travis build
    # @bentoml.api(FastaiImageHandler)
    # def predict_fastai_image(self, input_data):
    #     return self.artifacts.model.predict_image(input_data)
    #
    # @bentoml.api(FastaiImageHandler, input_names=('original', 'compared'))
    # def predict_fastai_images(self, original, compared):
    #     return all(original.data[0, 0] == compared.data[0, 0])
