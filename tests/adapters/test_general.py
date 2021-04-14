# pylint: disable=redefined-outer-name
import json

from bentoml.types import HTTPRequest


def test_cors(make_api):
    from bentoml.adapters import JsonInput, JsonOutput

    api = make_api(
        input_adapter=JsonInput(),
        output_adapter=JsonOutput(cors="*"),
        user_func=lambda i: i,
    )

    body = b'{"a": 1}'

    request = HTTPRequest(body=body)
    response = api.handle_request(request)

    assert response.body == body.decode()
    assert response.headers["Access-Control-Allow-Origin"] == "*"

    responses = api.handle_batch_request([request] * 3)
    for response in responses:
        assert response.body == body.decode()
        assert response.headers["Access-Control-Allow-Origin"] == "*"


def test_no_cors(make_api):
    from bentoml.adapters import JsonInput

    api = make_api(input_adapter=JsonInput(), user_func=lambda i: i)

    body = json.dumps("{a: 1}").encode('utf-8')

    request = HTTPRequest(body=body)
    response = api.handle_request(request)
    prediction = json.loads(response.body)
    assert "{a: 1}" == prediction
    assert "Access-Control-Allow-Origin" not in response.headers

    responses = api.handle_batch_request([request] * 3)
    for response in responses:
        prediction = json.loads(response.body)
        assert "{a: 1}" == prediction
        assert "Access-Control-Allow-Origin" not in response.headers
