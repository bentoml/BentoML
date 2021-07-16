import json
import pathlib
import sys
import time

import bentoml
from bentoml.adapters import FileInput, ImageInput, JsonInput, MultiImageInput
from bentoml.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
from bentoml.types import InferenceError, InferenceResult, InferenceTask


# pylint: disable=arguments-differ
@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PickleArtifact("model"), SklearnModelArtifact('sk_model')])
class NonBatchExampleService(bentoml.BentoService):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(
        input=MultiImageInput(input_names=('original', 'compared')), batch=False
    )
    def predict_multi_images(self, original, compared):
        return self.artifacts.model.predict_multi_images([original], [compared])[0]

    @bentoml.api(input=ImageInput(), batch=False)
    def predict_image(self, image):
        return self.artifacts.model.predict_image([image])[0]

    @bentoml.api(
        input=JsonInput(), mb_max_latency=1000, mb_max_batch_size=2000, batch=False
    )
    def predict_with_sklearn(self, json_value):
        return self.artifacts.sk_model.predict([json_value])[0]

    @bentoml.api(input=FileInput(), batch=False)
    def predict_file(self, file_):
        return self.artifacts.model.predict_file([file_])[0]

    @bentoml.api(input=JsonInput(), batch=False)
    def predict_json(self, input_data):
        return self.artifacts.model.predict_json([input_data])[0]

    @bentoml.api(input=JsonInput(), batch=False)
    def predict_strict_json(self, input_data, task: InferenceTask = None):
        if task.http_headers.content_type != "application/json":
            task.discard(http_status=400, err_msg="application/json only")
            return
        result = self.artifacts.model.predict_json([input_data])[0]
        return result

    @bentoml.api(input=JsonInput(), batch=False)
    def predict_direct_json(self, input_data, task: InferenceTask = None):
        if task.http_headers.content_type != "application/json":
            return InferenceError(http_status=400, err_msg="application/json only")
        result = self.artifacts.model.predict_json([input_data])[0]
        return InferenceResult(http_status=200, data=json.dumps(result))

    @bentoml.api(input=JsonInput(), mb_max_latency=10000 * 1000, batch=True)
    def echo_with_delay(self, input_datas):
        data = input_datas[0]
        time.sleep(data['b'] + data['a'] * len(input_datas))
        return input_datas

    @bentoml.api(input=JsonInput(), mb_max_latency=10000 * 1000, batch=True)
    def echo_batch_size(self, input_datas=10):
        data = input_datas[0]
        time.sleep(data['b'] + data['a'] * len(input_datas))
        batch_size = len(input_datas)
        return [batch_size] * batch_size


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    bento_dist_path = sys.argv[2]
    service = NonBatchExampleService()
    service.artifacts.load_all(artifacts_path)

    pathlib.Path(bento_dist_path).mkdir(parents=True, exist_ok=True)
    service.save_to_dir(bento_dist_path)
