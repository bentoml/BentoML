# pylint: disable=redefined-outer-name

import json

from bentoml.types import HTTPRequest


def test_string_input(make_api):
    from bentoml.adapters import JsonOutput, StringInput

    api = make_api(
        input_adapter=StringInput(),
        output_adapter=JsonOutput(),
        user_func=lambda i: i,
    )

    body = b'{"a": 1}'

    request = HTTPRequest(body=body)
    response = api.handle_request(request)

    assert json.loads(response.body) == body.decode()

    responses = api.handle_batch_request([request] * 3)
    for response in responses:
        assert json.loads(response.body) == body.decode()
