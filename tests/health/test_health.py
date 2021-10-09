from bentoml.health.health_server import BentoHealthServer


def test_api_function_route(bento_bundle_path):
    rest_server = BentoHealthServer()
    test_client = rest_server.app.test_client()

    response = test_client.get("/livez")
    assert 200 == response.status_code