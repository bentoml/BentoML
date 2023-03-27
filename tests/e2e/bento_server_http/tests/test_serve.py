import bentoml

def test_http_server(bentoml_home: str):
    server = bentoml.HTTPServer("service.py:svc", port=12345)

    server.start()
    client = server.get_client()
    resp = client.health()

    assert resp.status == 200

    res = client.echo_json_sync({"test": "json"})

    assert res == {"test": "json"}
