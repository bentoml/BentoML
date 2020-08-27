import base64
import mock

import flask
from urllib3.filepost import encode_multipart_formdata

from bentoml.adapters import LegacyImageInput


def predict(image):
    return image.shape


def test_image_input_cli(capsys, make_api, img_file):
    api = make_api(LegacyImageInput(input_names=("image",)), predict)
    test_args = ["--input-file-image", img_file]
    api.handle_cli(test_args)
    out, _ = capsys.readouterr()
    assert out.strip().endswith("(10, 10, 3)")


def test_image_input_aws_lambda_event(make_api, img_file):
    api = make_api(LegacyImageInput(input_names=("image",)), predict)
    with open(str(img_file), "rb") as image_file:
        content = image_file.read()
        try:
            image_bytes_encoded = base64.encodebytes(content)
        except AttributeError:
            image_bytes_encoded = base64.encodebytes(img_file)

    aws_lambda_event = {
        "body": image_bytes_encoded,
        "headers": {"Content-Type": "images/jpeg"},
    }

    aws_result = api.handle_aws_lambda_event(aws_lambda_event)
    assert aws_result["statusCode"] == 200
    assert aws_result["body"] == "[10, 10, 3]"


def test_image_input_http_request_post_binary(make_api, img_file):
    api = make_api(LegacyImageInput(input_names=("image",)), predict)
    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = open(str(img_file), 'rb').read()

    response = api.handle_request(request)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_image_input_http_request_multipart_form(make_api, img_file):
    api = make_api(LegacyImageInput(input_names=("image",)), predict)

    with open(img_file, "rb") as f:
        img_bytes = f.read()

    body, content_type = encode_multipart_formdata(dict(image=("test.jpg", img_bytes),))
    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.headers = {"Content-Type": content_type}
    request.get_data.return_value = body
    response = api.handle_request(request)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_image_input_http_request_single_image_different_name(make_api, img_file):
    api = make_api(LegacyImageInput(input_names=("image",)), predict)

    with open(img_file, "rb") as f:
        img_bytes = f.read()

    body, content_type = encode_multipart_formdata(
        dict(myimage=("test.jpg", img_bytes),)
    )
    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.headers = {"Content-Type": content_type}
    request.get_data.return_value = body
    response = api.handle_request(request)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_image_input_http_request_malformatted_input_missing_image_file(make_api,):
    api = make_api(LegacyImageInput(input_names=("image",)), predict)

    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.files = {}
    request.headers = {}
    request.get_data.return_value = None

    response = api.handle_request(request)

    assert response.status_code == 400
    assert response.data


def test_image_input_http_request_malformatted_input_wrong_input_name(
    make_api, img_file
):
    api = make_api(LegacyImageInput(input_names=("myimage", "myimage2")), predict)
    with open(img_file, "rb") as f:
        img_bytes = f.read()

    body, content_type = encode_multipart_formdata(
        dict(myimage=("test.jpg", img_bytes), myimage3=("test.jpg", img_bytes))
    )
    request = mock.MagicMock(spec=flask.Request)
    request.method = "POST"
    request.headers = {"Content-Type": content_type}
    request.get_data.return_value = body
    response = api.handle_request(request)

    assert response.status_code == 400
    assert response.data
