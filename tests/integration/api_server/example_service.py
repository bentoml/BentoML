import functools
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import bentoml
from bentoml.adapters import (  # FastaiImageInput,
    DataframeInput,
    FileInput,
    ImageInput,
    JsonInput,
    MultiImageInput,
)
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.handlers import DataframeHandler  # deprecated
from bentoml.saved_bundle import save_to_dir
from bentoml.service.artifacts.pickle import PickleArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PickleArtifact("model"), SklearnModelArtifact('sk_model')])
class ExampleBentoService(bentoml.BentoService):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(
        input=DataframeInput(dtype={"col1": "int"}),
        mb_max_latency=1000,
        mb_max_batch_size=2000,
        batch=True,
    )
    def predict_dataframe(self, df):
        return self.artifacts.model.predict_dataframe(df)

    @bentoml.api(DataframeHandler, dtype={"col1": "int"}, batch=True)  # deprecated
    def predict_dataframe_v1(self, df):
        return self.artifacts.model.predict_dataframe(df)

    @bentoml.api(
        input=MultiImageInput(input_names=('original', 'compared')), batch=True
    )
    def predict_legacy_images(self, originals, compareds):
        return self.artifacts.model.predict_multi_images(originals, compareds)

    @bentoml.api(input=ImageInput(), batch=True)
    def predict_image(self, images):
        return self.artifacts.model.predict_image(images)

    @bentoml.api(
        input=JsonInput(), mb_max_latency=1000, mb_max_batch_size=2000, batch=True,
    )
    def predict_with_sklearn(self, jsons):
        return self.artifacts.sk_model.predict(jsons)

    @bentoml.api(input=FileInput(), batch=True)
    def predict_file(self, files):
        return self.artifacts.model.predict_file(files)

    @bentoml.api(input=JsonInput(), batch=True)
    def predict_json(self, input_datas):
        return self.artifacts.model.predict_json(input_datas)

    @bentoml.api(input=JsonInput(), mb_max_latency=10000 * 1000, batch=True)
    def echo_with_delay(self, input_datas):
        data = input_datas[0]
        time.sleep(data['b'] + data['a'] * len(input_datas))
        return input_datas


# pylint: disable=arguments-differ
@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PickleArtifact("model"), SklearnModelArtifact('sk_model')])
class ExampleBentoServiceSingle(ExampleBentoService):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(
        input=MultiImageInput(input_names=('original', 'compared')), batch=False
    )
    def predict_legacy_images(self, original, compared):
        return self.artifacts.model.predict_multi_images(original, compared)

    @bentoml.api(input=ImageInput(), batch=False)
    def predict_image(self, image):
        return self.artifacts.model.predict_image([image])[0]

    @bentoml.api(
        input=JsonInput(), mb_max_latency=1000, mb_max_batch_size=2000, batch=False
    )
    def predict_with_sklearn(self, json):
        return self.artifacts.sk_model.predict([json])[0]

    @bentoml.api(input=FileInput(), batch=False)
    def predict_file(self, file_):
        return self.artifacts.model.predict_file([file_])[0]

    @bentoml.api(input=JsonInput(), batch=False)
    def predict_json(self, input_data):
        return self.artifacts.model.predict_json([input_data])[0]


class PickleModel(object):
    def predict_dataframe(self, df):
        return df['col1'] * 2

    def predict_image(self, input_datas):
        return [input_data.shape for input_data in input_datas]

    def predict_file(self, input_files):
        return [f.read() for f in input_files]

    def predict_multi_images(self, originals, compareds):
        return (np.array(originals) == np.array(compareds)).all()

    def predict_json(self, input_datas):
        return input_datas


@functools.lru_cache()
def gen_test_bundle(tmpdir, batch_mode=True):
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    if batch_mode:
        svc_cls = ExampleBentoService
    else:
        svc_cls = ExampleBentoServiceSingle
    svc_cls._bento_service_bundle_path = None
    svc_cls._bento_service_bundle_version = None
    test_svc = svc_cls()

    pickle_model = PickleModel()
    test_svc.pack('model', pickle_model)

    sklearn_model = RandomForestRegressor(n_estimators=2)
    sklearn_model.fit(
        [[i] for _ in range(100) for i in range(10)],
        [i for _ in range(100) for i in range(10)],
    )
    test_svc.pack('sk_model', sklearn_model)

    save_to_dir(test_svc, tmpdir, silent=True)
    return tmpdir
