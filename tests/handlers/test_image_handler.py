import base64
import glob
import pytest
import mock

import flask

from bentoml.exceptions import BadInput
from bentoml.adapters import ImageInput
from bentoml.marshal.utils import SimpleRequest


def predict(images):
    return [image.shape for image in images]


def test_image_input_cli(capsys, img_file):
    test_image_input = ImageInput()

    test_args = ["--input", img_file]
    test_image_input.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    assert out.strip().endswith("(10, 10, 3)")


def test_image_input_cli_list(capsys, img_files):
    test_image_input = ImageInput()

    test_args = ["--input"] + glob.glob(img_files)
    test_image_input.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    lines = out.strip().split('\n')
    for line in lines[-10:]:
        assert line.strip().endswith("(10, 10, 3)")


def test_image_input_aws_lambda_event(img_file):
    test_image_input = ImageInput()
    with open(str(img_file), "rb") as image_file:
        content = image_file.read()
        try:
            image_bytes_encoded = base64.encodebytes(content)
        except AttributeError:
            image_bytes_encoded = base64.encodebytes(str(img_file))

    aws_lambda_event = {
        "body": image_bytes_encoded,
        "headers": {"Content-Type": "images/png"},
    }

    aws_result = test_image_input.handle_aws_lambda_event(aws_lambda_event, predict)
    assert aws_result["statusCode"] == 200
    assert aws_result["body"] == "[10, 10, 3]"


def test_image_input_http_request_post_binary(img_file):
    test_image_input = ImageInput()
    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = open(str(img_file), 'rb').read()

    response = test_image_input.handle_request(request, predict)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)

    simple_request = SimpleRequest.from_flask_request(request)
    responses = test_image_input.handle_batch_request([simple_request], predict)

    assert responses[0].status == 200
    assert "[10, 10, 3]" in str(responses[0].data)


def test_image_input_http_request_multipart_form(img_file):
    test_image_input = ImageInput()
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(img_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': file_bytes,
        'stream': file_bytes,
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"image_file": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_image_input.handle_request(request, predict)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_image_input_http_request_single_image_different_name(img_file):
    test_image_input = ImageInput()
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(img_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': file_bytes,
        'stream': file_bytes,
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"a_differnt_name_used": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_image_input.handle_request(request, predict)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_image_input_http_request_malformatted_input_missing_image_file():
    test_image_input = ImageInput()
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_image_input.handle_request(request, predict)

    assert "unexpected HTTP request format" in str(e.value)


def test_image_input_http_request_malformatted_input_wrong_input_name():
    test_image_input = ImageInput()
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {"abc": None}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_image_input.handle_request(request, predict)

    assert "unexpected HTTP request format" in str(e.value)
