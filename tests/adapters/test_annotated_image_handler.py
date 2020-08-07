import base64
import glob
import io
import mock

import flask
import pytest

from urllib3.filepost import encode_multipart_formdata

from bentoml.exceptions import BadInput
from bentoml.adapters import AnnotatedImageInput
from bentoml.marshal.utils import SimpleRequest


def generate_multipart_body(image_file, json_file):
    image = ("image.jpg", open(image_file, "rb").read())
    json = ("annotations.json", open(json_file, "rb").read())

    files = {"image.jpg": image, "annotations.json": json}

    body, content_type = encode_multipart_formdata(files)

    headers = {
        'Content-Type': content_type,
        'Content-Length': len(body),
    }
    return body, headers

def predict_image_only(image):
    return image.shape


def predict_image_and_json(image, json_input):
    return (image.shape, json_input["name"])


def test_anno_image_input_cli_image_only(capsys, img_file):
    test_anno_image_input = AnnotatedImageInput()

    test_args = ["--image", img_file]
    test_anno_image_input.handle_cli(test_args, predict_image_only)
    out, _ = capsys.readouterr()

    assert out.strip() == "(10, 10, 3)"


def test_anno_image_input_cli_image_and_json(capsys, img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()

    test_args = ["--image", img_file, "--json", json_file]
    test_anno_image_input.handle_cli(test_args, predict_image_and_json)
    out, _ = capsys.readouterr()

    assert out.strip() == "((10, 10, 3), 'kaith')"


def test_anno_image_input_aws_lambda_event(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()

    multipart_data, headers = generate_multipart_body(img_file, json_file)

    aws_lambda_event = {"body": multipart_data, "headers": headers}
    aws_result = test_anno_image_input.handle_aws_lambda_event(
            aws_lambda_event, predict_image_and_json)

    assert aws_result["statusCode"] == 200
    assert aws_result["body"] == '[[10, 10, 3], "kaith"]'


def test_anno_image_input_http_request_multipart_form(img_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(img_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': file_bytes,
        'mimetype': 'image/png',
        'stream': io.BytesIO(file_bytes),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"image_file": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_anno_image_input.handle_request(request, predict_image_only)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_anno_image_input_http_request_single_image_different_name(img_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(img_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': file_bytes,
        'mimetype': 'image/png',
        'stream': io.BytesIO(file_bytes),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"a_differnt_name_used": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_anno_image_input.handle_request(request, predict_image_only)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_anno_image_input_http_request_malformatted_input_missing_image_file():
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_only)

    assert "unexpected HTTP request format" in str(e.value)


def test_anno_image_input_http_request_malformatted_input_wrong_input_name():
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {"abc": None}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_only)

    assert "unexpected HTTP request format" in str(e.value)


# TODO: Add tests for handle_batch_request

