import base64
import pytest
import mock

import flask

from bentoml.exceptions import BadInput
from bentoml.adapters import LegacyImageInput as ImageInput


def predict(image):
    return image.shape


def test_image_input_cli(capsys, img_file):
    test_image_input = ImageInput()

    test_args = ["--input={}".format(img_file)]
    test_image_input.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    assert out.strip().endswith("(10, 10, 3)")


def test_image_input_aws_lambda_event(img_file):
    test_image_input = ImageInput()
    with open(str(img_file), "rb") as image_file:
        content = image_file.read()
        try:
            image_bytes_encoded = base64.encodebytes(content)
        except AttributeError:
            image_bytes_encoded = base64.encodebytes(img_file)

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
    request.get_data.return_value = open(str(img_file), 'rb')

    response = test_image_input.handle_request(request, predict)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_image_input_http_request_multipart_form(img_file):
    test_image_input = ImageInput(input_names=("my_image",))
    request = mock.MagicMock(spec=flask.Request)
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': open(str(img_file), 'rb').read(),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"my_image": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_image_input.handle_request(request, predict)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_image_input_http_request_single_image_different_name(img_file):
    test_image_input = ImageInput(input_names=("my_image",))
    request = mock.MagicMock(spec=flask.Request)
    file_attr = {
        'filename': 'test_img.png',
        'read.return_value': open(str(img_file), 'rb').read(),
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
    test_image_input = ImageInput(input_names=("my_image",))
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_image_input.handle_request(request, predict)

    assert "unexpected HTTP request format" in str(e.value)


def test_image_input_http_request_malformatted_input_wrong_input_name():
    test_image_input = ImageInput(input_names=("my_image", "my_image2"))
    request = mock.MagicMock(spec=flask.Request)

    request.method = "POST"
    request.files = {"abc": None}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_image_input.handle_request(request, predict)

    assert "unexpected HTTP request format" in str(e.value)
