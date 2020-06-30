import sys

from sklearn.ensemble import RandomForestRegressor

import bentoml
from bentoml.saved_bundle import save_to_dir
from bentoml.adapters import (
    DataframeInput,
    ImageInput,
    LegacyImageInput,
    JsonInput,
    # FastaiImageInput,
)
from bentoml.handlers import DataframeHandler  # deprecated
from bentoml.artifact import PickleArtifact, SklearnModelArtifact


@bentoml.artifacts([PickleArtifact("model"), SklearnModelArtifact('sk_model')])
@bentoml.env(auto_pip_dependencies=True)
class ExampleBentoService(bentoml.BentoService):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(
        input=JsonInput(), mb_max_latency=1000, mb_max_batch_size=2000,
    )
    def predict_with_sklearn(self, jsons):
        """predict_dataframe expects dataframe as input
        """
        return self.artifacts.sk_model.predict(jsons)

    @bentoml.api(
        input=DataframeInput(input_dtypes={"col1": "int"}),
        mb_max_latency=1000,
        mb_max_batch_size=2000,
    )
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
    def predict_legacy_images(self, original, compared):
        return self.artifacts.model.predict_legacy_images(original, compared)

    @bentoml.api(input=JsonInput())
    def predict_json(self, input_data):
        return self.artifacts.model.predict_json(input_data)


class PickleModel(object):
    def predict_dataframe(self, df):
        return df["col1"] * 2

    def predict_image(self, input_datas):
        for input_data in input_datas:
            assert input_data is not None
        return [input_data.shape for input_data in input_datas]

    def predict_legacy_images(self, original, compared):
        return (original == compared).all()

    def predict_json(self, input_datas):
        assert input_datas is not None
        return input_datas


if __name__ == "__main__":
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    ExampleBentoService._bento_service_bundle_path = None
    ExampleBentoService._bento_service_bundle_version = None
    test_svc = ExampleBentoService()

    pickle_model = PickleModel()
    test_svc.pack('model', pickle_model)

    sklearn_model = RandomForestRegressor(n_estimators=2)
    sklearn_model.fit([[i] for i in range(10000)], [i for i in range(10000)])
    test_svc.pack('sk_model', sklearn_model)

    tmpdir = sys.argv[1]
    # version = sys.argv[2]
    save_to_dir(test_svc, tmpdir, silent=True)
