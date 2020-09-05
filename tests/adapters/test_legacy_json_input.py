import flask
import mock

from bentoml.adapters import LegacyJsonInput


def predict(obj):
    return obj["name"]


def test_json_handle_cli(capsys, make_api, json_file):
    api = make_api(LegacyJsonInput(), predict)

    test_args = ["--input-file", json_file]
    api.handle_cli(test_args)
    out, _ = capsys.readouterr()
    assert out.strip() == '"kaith"'


def test_json_handle_aws_lambda_event(make_api, json_file):
    with open(json_file) as f:
        test_content = f.read()
    api = make_api(LegacyJsonInput(), predict)

    success_event_obj = {
        "headers": {"Content-Type": "application/json"},
        "body": test_content,
    }
    success_response = api.handle_aws_lambda_event(success_event_obj)

    assert success_response["statusCode"] == 200
    assert success_response["body"] == '"kaith"'

    error_event_obj = {
        "headers": {"Content-Type": "application/json"},
        "body": "bad json{}",
    }
    error_response = api.handle_aws_lambda_event(error_event_obj)
    assert error_response["statusCode"] == 400
    assert error_response["body"]


def test_image_input_http_request_post_json(make_api, json_file):
    api = make_api(LegacyJsonInput(), predict)
    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = open(str(json_file), 'rb').read()

    response = api.handle_request(request)

    assert response.status_code == 200
    assert response.json == 'kaith'
