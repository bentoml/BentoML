import json
import pytest

import flask

from bentoml.adapters import DataframeInput
from bentoml.types import InferenceResult

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


@pytest.fixture(name='inference_result_api')
def api_returns_inference_result(make_api):
    def test_func_inference_result(df):
        return [
            InferenceResult(
                data=json.dumps({'name': df['name'][0]}),
                http_status=200,
                http_headers={
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
            )
        ]

    api = make_api(DataframeInput(), test_func_inference_result)
    return api


def test_default_output_http_requests(inference_result_api):
    csv_data = b'name,game,city\njohn,mario,sf'
    request = MagicMock(spec=flask.Request)
    request.headers = {'Content-Type': 'text/csv'}
    request.get_data.return_value = csv_data

    result = inference_result_api.handle_request(request)
    assert result.get_data().decode('utf-8') == '{"name": "john"}'
    assert result.headers['Content-Type'] == 'application/json'
    assert result.headers['Access-Control-Allow-Origin'] == '*'


def test_default_output_cli(capsys, inference_result_api, tmpdir):
    json_file = tmpdir.join("test.csv")
    with open(str(json_file), "w") as f:
        f.write('name,game,city\njohn,mario,sf')

    test_args = ["--input-file", str(json_file), "--format", "csv"]
    inference_result_api.handle_cli(test_args)
    out, _ = capsys.readouterr()
    assert "john" in out


def test_default_output_aws_lambda(inference_result_api):
    test_content = '[{"name": "john","game": "mario","city": "sf"}]'

    event_without_content_type_header = {
        "headers": {},
        "body": test_content,
    }
    response = inference_result_api.handle_aws_lambda_event(
        event_without_content_type_header
    )
    assert response["statusCode"] == 200
    assert response["body"] == '{"name": "john"}'
