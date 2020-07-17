import os
import json
from io import BytesIO

from bentoml.server.api_server import BentoAPIServer

CUR_PATH = os.path.dirname(os.path.abspath(__file__))


def test_api_function_route(bento_service, tmpdir, img_file):
    import imageio  # noqa # pylint: disable=unused-import
    import numpy as np  # noqa # pylint: disable=unused-import

    rest_server = BentoAPIServer(bento_service)
    test_client = rest_server.app.test_client()

    index_list = []
    for rule in rest_server.app.url_map.iter_rules():
        index_list.append(rule.endpoint)

    response = test_client.get("/")
    assert 200 == response.status_code

    response = test_client.get("/healthz")
    assert 200 == response.status_code

    response = test_client.get("/docs.json")
    assert 200 == response.status_code

    assert "predict_dataframe" in index_list
    data = [{"col1": 10}, {"col1": 20}]
    response = test_client.post(
        "/predict_dataframe", data=json.dumps(data), content_type="application/json"
    )
    assert response.data.decode().strip() == '30'

    assert "predict_dataframe_v1" in index_list
    data = [{"col1": 10}, {"col1": 20}]
    response = test_client.post(
        "/predict_dataframe_v1", data=json.dumps(data), content_type="application/json"
    )
    assert response.data.decode().strip() == '30'

    # Test ImageInput.
    with open(str(img_file), "rb") as f:
        img = f.read()

    response = test_client.post(
        "/predict_image", data={'image': (BytesIO(img), 'test_img.png')}
    )
    assert 200 == response.status_code
    assert "[10, 10, 3]" in str(response.data)

    response = test_client.post(
        "/predict_legacy_images",
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
