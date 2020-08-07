import base64
import glob
import io
import pytest
import mock

import flask

from bentoml.exceptions import BadInput
from bentoml.adapters import FileInput
from bentoml.marshal.utils import SimpleRequest


def predict(files):
    return [file.read() for file in files]


def test_file_input_cli(capsys, bin_file):
    test_file_input = FileInput()

    test_args = ["--input", bin_file]
    test_file_input.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    assert '\\x810\\x899' in out


def test_file_input_cli_list(capsys, bin_files):
    test_file_input = FileInput()

    test_args = ["--input"] + glob.glob(bin_files)
    test_file_input.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    lines = out.strip().split('\n')
    for line in lines[-10:]:
        assert '\\x810\\x899' in line


def test_file_input_aws_lambda_event(bin_file):
    test_file_input = FileInput()
    with open(str(bin_file), "rb") as file_file:
        content = file_file.read()
        try:
            file_bytes_encoded = base64.encodebytes(content)
        except AttributeError:
            file_bytes_encoded = base64.encodebytes(str(bin_file))

    aws_lambda_event = {
        "body": file_bytes_encoded,
        "headers": {"Content-Type": "images/png"},
    }

    aws_result = test_file_input.handle_aws_lambda_event(aws_lambda_event, predict)
    assert aws_result["statusCode"] == 200
    assert aws_result["body"] == '{"b64": "gTCJOQ=="}'


def test_file_input_http_request_post_binary(bin_file):
    test_file_input = FileInput()
    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = open(str(bin_file), 'rb').read()

    response = test_file_input.handle_request(request)

    assert response.status_code == 200
    assert b'{"b64": "gTCJOQ=="}' in response.data

    simple_request = SimpleRequest.from_flask_request(request)
    responses = test_file_input.handle_batch_request([simple_request], predict)

    assert responses[0].status == 200
    assert '{"b64": "gTCJOQ=="}' == responses[0].data


def test_file_input_http_request_multipart_form(bin_file):
    test_file_input = FileInput()
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(bin_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': file_bytes,
        'stream': io.BytesIO(file_bytes),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"file_file": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_file_input.handle_request(request)

    assert response.status_code == 200
    assert b'{"b64": "gTCJOQ=="}' in response.data


def test_file_input_http_request_single_file_different_name(bin_file):
    test_file_input = FileInput()
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(bin_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': file_bytes,
        'stream': io.BytesIO(file_bytes),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"a_differnt_name_used": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_file_input.handle_request(request)

    assert response.status_code == 200
    assert b'{"b64": "gTCJOQ=="}' in response.data


def test_file_input_http_request_malformatted_input_missing_file_file():
    test_file_input = FileInput()
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_file_input.handle_request(request)

    assert "unexpected HTTP request format" in str(e.value)


def test_file_input_http_request_malformatted_input_wrong_input_name():
    test_file_input = FileInput()
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {"abc": None}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_file_input.handle_request(request)

    assert "unexpected HTTP request format" in str(e.value)
