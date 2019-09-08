import os
import sys
import json
from io import BytesIO


from bentoml.server import BentoAPIServer

CUR_PATH = os.path.dirname(os.path.abspath(__file__))


def test_api_function_route(bento_service):
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

    assert "predict" in index_list
    data = [{"age": 10}]

    response = test_client.post(
        "/predict", data=json.dumps(data), content_type="application/json"
    )

    response_data = json.loads(response.data)
    assert 15 == response_data[0]["age"]

    # Test Image handlers.
    with open(os.path.join(CUR_PATH, "white-plane-sky.jpg"), "rb") as f:
        img = f.read()

    response = test_client.post("/predictImage", data=img, content_type="image/png")
    assert 200 == response.status_code

    response = test_client.post(
        "/predictImages",
        data={
            'original': (BytesIO(img), 'original.jpg'),
            'compared': (BytesIO(img), 'compared.jpg'),
        },
    )
    assert 200 == response.status_code

    # Test Fastai Image Handlers.
    if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
        # fast ai is required 3.6 or higher.
        response = test_client.post(
            "/predictFastaiImage", data=img, content_type="image/png"
        )
        assert 200 == response.status_code

        response = test_client.post(
            "/predictFastaiImages",
            data={
                'original': (BytesIO(img), 'original.jpg'),
                'compared': (BytesIO(img), 'compared.jpg'),
            },
        )
        assert 200 == response.status_code
