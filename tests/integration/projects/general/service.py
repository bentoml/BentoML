import json
import pathlib
import sys
import time
from typing import Sequence

from packaging import version

import bentoml
from bentoml import __version__ as BENTOML_VERSION
from bentoml.adapters import (
    DataframeInput,
    FileInput,
    ImageInput,
    JsonInput,
    JsonOutput,
    MultiImageInput,
)
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.handlers import DataframeHandler
from bentoml.service.artifacts.pickle import PickleArtifact
from bentoml.types import InferenceResult, InferenceTask


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PickleArtifact("model"), SklearnModelArtifact('sk_model')])
class ExampleService(bentoml.BentoService):
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
    def predict_multi_images(self, originals, compareds):
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

    CUSTOM_ROUTE = "$~!@%^&*()_-+=[]\\|;:,./predict"

    @bentoml.api(
        route=CUSTOM_ROUTE, input=JsonInput(), batch=True,
    )
    def customezed_route(self, input_datas):
        return input_datas

    CUSTOM_SCHEMA = {
        "application/json": {
            "schema": {
                "type": "object",
                "required": ["field1", "field2"],
                "properties": {
                    "field1": {"type": "string"},
                    "field2": {"type": "uuid"},
                },
            },
        }
    }

    @bentoml.api(input=JsonInput(request_schema=CUSTOM_SCHEMA), batch=True)
    def customezed_schema(self, input_datas):
        return input_datas

    @bentoml.api(input=JsonInput(), batch=True)
    def predict_strict_json(self, input_datas, tasks: Sequence[InferenceTask] = None):
        filtered_jsons = []
        for j, t in zip(input_datas, tasks):
            if t.http_headers.content_type != "application/json":
                t.discard(http_status=400, err_msg="application/json only")
            else:
                filtered_jsons.append(j)
        return self.artifacts.model.predict_json(filtered_jsons)

    @bentoml.api(input=JsonInput(), batch=True)
    def predict_direct_json(self, input_datas, tasks: Sequence[InferenceTask] = None):
        filtered_jsons = []
        for j, t in zip(input_datas, tasks):
            if t.http_headers.content_type != "application/json":
                t.discard(http_status=400, err_msg="application/json only")
            else:
                filtered_jsons.append(j)
        rets = self.artifacts.model.predict_json(filtered_jsons)
        return [
            InferenceResult(http_status=200, data=json.dumps(result)) for result in rets
        ]

    @bentoml.api(input=JsonInput(), mb_max_latency=10000 * 1000, batch=True)
    def echo_with_delay(self, input_datas):
        data = input_datas[0]
        time.sleep(data['b'] + data['a'] * len(input_datas))
        return input_datas

    @bentoml.api(input=JsonInput(), mb_max_batch_size=1, batch=True)
    def echo_batch_size_1(self, input_datas):
        data = input_datas[0]
        time.sleep(data['b'] + data['a'] * len(input_datas))
        batch_size = len(input_datas)
        return [batch_size] * batch_size

    @bentoml.api(input=JsonInput(), mb_max_latency=10000 * 1000, batch=True)
    def echo_batch_size(self, input_datas=10):
        data = input_datas[0]
        time.sleep(data['b'] + data['a'] * len(input_datas))
        batch_size = len(input_datas)
        return [batch_size] * batch_size

    @bentoml.api(input=JsonInput())
    def echo_json(self, input_data):
        return input_data

    if version.parse(BENTOML_VERSION) > version.parse("0.12.1+0"):

        @bentoml.api(input=JsonInput(), output=JsonOutput(ensure_ascii=True))
        def echo_json_ensure_ascii(self, input_data):
            return input_data


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    bento_dist_path = sys.argv[2]
    service = ExampleService()
    service.artifacts.load_all(artifacts_path)

    pathlib.Path(bento_dist_path).mkdir(parents=True, exist_ok=True)
    service.save_to_dir(bento_dist_path)
