import json

from bentoml.server import BentoAPIServer


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
