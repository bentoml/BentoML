from io import BytesIO
import json
import os

import simple_di

from bentoml.configuration.containers import BentoMLContainer
from bentoml.server.model_app import ModelApp


CUR_PATH = os.path.dirname(os.path.abspath(__file__))

CUSTOM_ROUTE = "$~!@%^&*()_-+=[]\\|;:,./predict"


def test_api_function_route(bento_bundle_path, img_file):
    import imageio  # noqa # pylint: disable=unused-import
    import numpy as np  # noqa # pylint: disable=unused-import

    BentoMLContainer.metrics_client._cache = simple_di.sentinel  # reset metrics_client
    rest_server = ModelApp(
        bundle_path=bento_bundle_path,
        enable_swagger=True,
        enable_metrics=True,
        enable_feedback=True,
    )
    test_client = rest_server.app.test_client()

    index_list = []
    for rule in rest_server.app.url_map.iter_rules():
        index_list.append(rule.endpoint)

    response = test_client.get("/")
    assert 200 == response.status_code

    response = test_client.get("/healthz")
    assert 200 == response.status_code

    response = test_client.get("/metadata")
    assert 200 == response.status_code

    response = test_client.get("/docs.json")
    assert 200 == response.status_code
    docs = json.loads(response.data.decode())
    assert f"/{CUSTOM_ROUTE}" in docs["paths"]

    response = test_client.post(f"/{CUSTOM_ROUTE}", data='{"a": 1}',)
    assert 200 == response.status_code
    assert '{"a": 1}' == response.data.decode()

    assert "predict_dataframe" in index_list
    data = [{"col1": 10}, {"col1": 20}]
    response = test_client.post(
        "/predict_dataframe", data=json.dumps(data), content_type="application/json"
    )
    assert response.data.decode().strip() == '[{"col1":20},{"col1":40}]'

    assert "predict_dataframe_v1" in index_list
    data = [{"col1": 10}, {"col1": 20}]
    response = test_client.post(
        "/predict_dataframe_v1", data=json.dumps(data), content_type="application/json"
    )
    assert response.data.decode().strip() == '[{"col1":20},{"col1":40}]'

    # Test ImageInput.
    with open(str(img_file), "rb") as f:
        img = f.read()

    response = test_client.post(
        "/predict_image", data={'image': (BytesIO(img), 'test_img.png')}
    )
    assert 200 == response.status_code
    assert "[10, 10, 3]" in str(response.data)

    response = test_client.post(
        "/predict_multi_images",
        data={
            'original': (BytesIO(img), 'original.jpg'),
            'compared': (BytesIO(img), 'compared.jpg'),
        },
    )
    assert 200 == response.status_code

    # Disabling fastai related tests to fix travis build
    # response = test_client.post(
    #     "/predict_fastai_image", data=img, content_type="image/png"
    # )
    # assert 200 == response.status_code
    #
    # response = test_client.post(
    #     "/predict_fastai_images",
    #     data={
    #         'original': (BytesIO(img), 'original.jpg'),
    #         'compared': (BytesIO(img), 'compared.jpg'),
    #     },
    # )
    # assert 200 == response.status_code


def test_api_function_route_with_disabled_swagger(bento_bundle_path):
    BentoMLContainer.metrics_client._cache = simple_di.sentinel  # reset metrics_client
    rest_server = ModelApp(
        bundle_path=bento_bundle_path,
        enable_swagger=False,
        enable_metrics=True,
        enable_feedback=True,
    )
    test_client = rest_server.app.test_client()

    response = test_client.get("/")
    assert 404 == response.status_code

    response = test_client.get("/docs")
    assert 404 == response.status_code

    response = test_client.get("/healthz")
    assert 200 == response.status_code

    response = test_client.get("/docs.json")
    assert 200 == response.status_code
