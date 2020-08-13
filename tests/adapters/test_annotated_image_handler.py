import io
import os
import mock

import flask
import pytest

from werkzeug import Request
from urllib3.filepost import encode_multipart_formdata

from bentoml.exceptions import BadInput
from bentoml.adapters import AnnotatedImageInput
from bentoml.marshal.utils import SimpleRequest


def generate_multipart_body(image_file, json_file=None):
    image = ("image.jpg", open(image_file, "rb").read())
    files = {"image.jpg": image}

    if json_file:
        json = ("annotations.json", open(json_file, "rb").read())
        files["annotations.json"] = json

    body, content_type = encode_multipart_formdata(files)

    headers = {
        'Content-Type': content_type,
        'Content-Length': len(body),
    }
    return body, headers


def predict_image_only(image):
    return image.shape


def predict_image_and_json(image, annotations):
    return (image.shape, annotations["name"])


def test_anno_image_input_cli_image_only(capsys, img_file):
    test_anno_image_input = AnnotatedImageInput()

    test_args = ["--image", img_file]
    test_anno_image_input.handle_cli(test_args, predict_image_only)
    out, _ = capsys.readouterr()

    assert out.strip() == "(10, 10, 3)"


def test_anno_image_input_cli_image_and_json(capsys, img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()

    test_args = ["--image", img_file, "--annotations", json_file]
    test_anno_image_input.handle_cli(test_args, predict_image_and_json)
    out, _ = capsys.readouterr()

    assert out.strip() == "((10, 10, 3), 'kaith')"


def test_anno_image_input_cli_relative_paths(capsys, img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()

    try:
        # This fails on Windows if our file is on a different drive
        relative_image_path = os.path.relpath(img_file)
        relative_annotation_path = os.path.relpath(json_file)
    except ValueError:
        # Switch to the drive with the files image on it, and try again
        os.chdir(img_file[:2])
        relative_image_path = os.path.relpath(img_file)
        relative_annotation_path = os.path.relpath(json_file)

    test_args = [
        "--image",
        relative_image_path,
        "--annotations",
        relative_annotation_path,
    ]
    test_anno_image_input.handle_cli(test_args, predict_image_and_json)
    out, _ = capsys.readouterr()

    assert out.strip() == "((10, 10, 3), 'kaith')"


def test_anno_image_input_aws_lambda_event(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()

    multipart_data, headers = generate_multipart_body(img_file, json_file)

    aws_lambda_event = {"body": multipart_data, "headers": headers}
    aws_result = test_anno_image_input.handle_aws_lambda_event(
        aws_lambda_event, predict_image_and_json
    )

    assert aws_result["statusCode"] == 200
    assert aws_result["body"] == '[[10, 10, 3], "kaith"]'


def test_anno_image_input_aws_lambda_event_bad_content_type(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()

    multipart_data, headers = generate_multipart_body(img_file, json_file)
    headers['Content-Type'] = 'image/jpeg'

    aws_lambda_event = {"body": multipart_data, "headers": headers}

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_aws_lambda_event(
            aws_lambda_event, predict_image_and_json
        )

    assert "only supports multipart/form-data" in str(e.value)


def test_anno_image_input_http_request_multipart_form(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    image_file_bytes = open(str(img_file), 'rb').read()
    image_file_attr = {
        'filename': 'test_img.png',
        'read.return_value': image_file_bytes,
        'mimetype': 'image/png',
        'stream': io.BytesIO(image_file_bytes),
    }
    image_file = mock.Mock(**image_file_attr)

    json_file_bytes = open(str(json_file), 'rb').read()
    json_file_attr = {
        'filename': 'annotations.json',
        'read.return_value': json_file_bytes,
        'mimetype': 'application/json',
        'stream': io.BytesIO(json_file_bytes),
    }
    json_file = mock.Mock(**json_file_attr)

    request.method = "POST"
    request.files = {"image_file": image_file, "json_file": json_file}

    request.get_data.return_value = None

    response = test_anno_image_input.handle_request(request, predict_image_and_json)

    assert response.status_code == 200
    assert '[10, 10, 3], "kaith"' in str(response.response)


def test_anno_image_input_http_request_invalid_json(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    image_file_bytes = open(str(img_file), 'rb').read()
    image_file_attr = {
        'filename': 'test_img.png',
        'read.return_value': image_file_bytes,
        'mimetype': 'image/png',
        'stream': io.BytesIO(image_file_bytes),
    }
    image_file = mock.Mock(**image_file_attr)

    json_file_bytes = open(str(json_file), 'rb').read()
    json_file_bytes = json_file_bytes[: int(len(json_file_bytes) / 2)]
    json_file_attr = {
        'filename': 'annotations.json',
        'read.return_value': json_file_bytes,
        'mimetype': 'application/json',
        'stream': io.BytesIO(json_file_bytes),
    }
    json_file = mock.Mock(**json_file_attr)

    request.method = "POST"
    request.files = {"image_file": image_file, "json_file": json_file}

    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_and_json)

    assert "invalid JSON" in str(e.value)


def test_anno_image_input_http_request_multipart_form_image_only(img_file):
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


def test_anno_image_input_http_request_too_many_files(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    image_file_bytes = open(str(img_file), 'rb').read()
    image_file_attr = {
        'filename': 'test_img.png',
        'read.return_value': image_file_bytes,
        'mimetype': 'image/png',
        'stream': io.BytesIO(image_file_bytes),
    }
    image_file = mock.Mock(**image_file_attr)

    json_file_bytes = open(str(json_file), 'rb').read()
    json_file_attr = {
        'filename': 'annotations.json',
        'read.return_value': json_file_bytes,
        'mimetype': 'application/json',
        'stream': io.BytesIO(json_file_bytes),
    }
    json_file = mock.Mock(**json_file_attr)

    request.method = "POST"
    request.files = {
        "image_file": image_file,
        "json_file": json_file,
        "image_file_2": image_file,
    }
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_and_json)

    assert "Too many input files" in str(e.value)


def test_anno_image_input_http_request_two_image_files(img_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    image_file_bytes = open(str(img_file), 'rb').read()
    image_file_attr = {
        'filename': 'test_img.png',
        'read.return_value': image_file_bytes,
        'mimetype': 'image/png',
        'stream': io.BytesIO(image_file_bytes),
    }
    image_file = mock.Mock(**image_file_attr)

    request.method = "POST"
    request.files = {"image_file": image_file, "image_file_2": image_file}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_only)

    assert "received two images" in str(e.value)


def test_anno_image_input_http_request_two_json_files(json_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    json_file_bytes = open(str(json_file), 'rb').read()
    json_file_attr = {
        'filename': 'annotations.json',
        'read.return_value': json_file_bytes,
        'mimetype': 'application/json',
        'stream': io.BytesIO(json_file_bytes),
    }
    json_file = mock.Mock(**json_file_attr)

    request.method = "POST"
    request.files = {"json_file": json_file, "json_file_2": json_file}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_only)

    assert "received two JSON files" in str(e.value)


def test_anno_image_input_http_request_only_json_file(json_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    json_file_bytes = open(str(json_file), 'rb').read()
    json_file_attr = {
        'filename': 'annotations.json',
        'read.return_value': json_file_bytes,
        'mimetype': 'application/json',
        'stream': io.BytesIO(json_file_bytes),
    }
    json_file = mock.Mock(**json_file_attr)

    request.method = "POST"
    request.files = {"json_file": json_file}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_only)

    assert "requires an image file" in str(e.value)


def test_anno_image_input_batch_request(img_file, json_file):
    adapter = AnnotatedImageInput(is_batch_input=True)

    multipart_data, headers = generate_multipart_body(img_file, json_file)
    request = SimpleRequest.from_flask_request(
        Request.from_values(
            data=multipart_data,
            content_type=headers['Content-Type'],
            content_length=headers['Content-Length'],
        )
    )

    responses = adapter.handle_batch_request([request] * 5, predict_image_and_json)
    for response in responses:
        assert response.status == 200
        assert response.data == '[[10, 10, 3], "kaith"]'


def test_anno_image_input_check_config():
    adapter = AnnotatedImageInput()
    config = adapter.config
    assert isinstance(config["accept_image_formats"], list) and isinstance(
        config["pilmode"], str
    )


def test_anno_image_input_check_request_schema():
    adapter = AnnotatedImageInput()
    assert isinstance(adapter.request_schema, dict)


def test_anno_image_input_check_pip_deps():
    adapter = AnnotatedImageInput()
    assert isinstance(adapter.pip_dependencies, list)


def test_anno_image_input_batch_request_skip_bad(img_file, json_file):
    adapter = AnnotatedImageInput(is_batch_input=True)

    multipart_data, headers = generate_multipart_body(img_file, json_file)

    empty_request = SimpleRequest(headers=headers, data=None)

    request = SimpleRequest.from_flask_request(
        Request.from_values(
            data=multipart_data,
            content_type=headers['Content-Type'],
            content_length=headers['Content-Length'],
        )
    )

    image = ("image.jpg", open(img_file, "rb").read())
    json = ("annotations.jso", open(json_file, "rb").read())
    files = {"image.invalid": image, "annotations.invalid": json}
    bad_data, content_type = encode_multipart_formdata(files)

    bad_request = SimpleRequest.from_flask_request(
        Request.from_values(
            data=bad_data, content_type=content_type, content_length=len(bad_data),
        )
    )

    responses = adapter.handle_batch_request(
        [empty_request, request, bad_request], predict_image_and_json
    )

    assert len(responses) == 3
    assert responses[0] is None
    assert responses[1].status == 200 and responses[1].data == '[[10, 10, 3], "kaith"]'
    assert responses[2] is None

    bad_responses = adapter.handle_batch_request(
        [empty_request], predict_image_and_json
    )
    assert len(bad_responses) == 1
    assert bad_responses[0] is None


def test_anno_image_input_batch_request_image_only(img_file):
    adapter = AnnotatedImageInput(is_batch_input=True)

    multipart_data, headers = generate_multipart_body(img_file)
    request = SimpleRequest.from_flask_request(
        Request.from_values(
            data=multipart_data,
            content_type=headers['Content-Type'],
            content_length=headers['Content-Length'],
        )
    )

    responses = adapter.handle_batch_request([request] * 5, predict_image_only)
    for response in responses:
        assert response.status == 200
        assert response.data == '[10, 10, 3]'


def test_anno_image_input_http_request_multipart_octet_streams(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    image_file_bytes = open(str(img_file), 'rb').read()
    image_file_attr = {
        'filename': 'test_img.png',
        'read.return_value': image_file_bytes,
        'mimetype': 'application/octet-stream',
        'stream': io.BytesIO(image_file_bytes),
    }
    image_file = mock.Mock(**image_file_attr)

    json_file_bytes = open(str(json_file), 'rb').read()
    json_file_attr = {
        'filename': 'annotations.json',
        'read.return_value': json_file_bytes,
        'mimetype': 'application/octet-stream',
        'stream': io.BytesIO(json_file_bytes),
    }
    json_file = mock.Mock(**json_file_attr)

    request.method = "POST"
    request.files = {"image_file": image_file, "json_file": json_file}

    request.get_data.return_value = None

    response = test_anno_image_input.handle_request(request, predict_image_and_json)

    assert response.status_code == 200
    assert '[10, 10, 3], "kaith"' in str(response.response)


def test_anno_image_input_octet_stream_custom_image_extension(img_file):
    test_anno_image_input = AnnotatedImageInput(accept_image_formats=[".custom"])
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(img_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.custom',
        'read.return_value': file_bytes,
        'mimetype': 'application/octet-stream',
        'stream': io.BytesIO(file_bytes),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"test_img": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_anno_image_input.handle_request(request, predict_image_only)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_anno_image_input_custom_accept_extension_not_accepted(img_file):
    test_anno_image_input = AnnotatedImageInput(accept_image_formats=[".custom"])
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(img_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.jpg',
        'read.return_value': file_bytes,
        'mimetype': 'image/jpg',
        'stream': io.BytesIO(file_bytes),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"test_img": file}
    request.headers = {}
    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_only)

    assert "Input file not in supported format list" in str(e.value)


def test_anno_image_input_octet_stream_json(img_file):
    test_anno_image_input = AnnotatedImageInput(accept_image_formats=[".custom"])
    request = mock.MagicMock(spec=flask.Request)
    file_bytes = open(str(img_file), 'rb').read()
    file_attr = {
        'filename': 'test_img.custom',
        'read.return_value': file_bytes,
        'mimetype': 'application/octet-stream',
        'stream': io.BytesIO(file_bytes),
    }
    file = mock.Mock(**file_attr)

    request.method = "POST"
    request.files = {"a_different_name_used": file}
    request.headers = {}
    request.get_data.return_value = None

    response = test_anno_image_input.handle_request(request, predict_image_only)

    assert response.status_code == 200
    assert "[10, 10, 3]" in str(response.response)


def test_anno_image_input_octet_stream_bad_json_filename(img_file, json_file):
    test_anno_image_input = AnnotatedImageInput()
    request = mock.MagicMock(spec=flask.Request)
    image_file_bytes = open(str(img_file), 'rb').read()
    image_file_attr = {
        'filename': 'test_img.png',
        'read.return_value': image_file_bytes,
        'mimetype': 'application/octet-stream',
        'stream': io.BytesIO(image_file_bytes),
    }
    image_file = mock.Mock(**image_file_attr)

    json_file_bytes = open(str(json_file), 'rb').read()
    json_file_attr = {
        'filename': 'annotations.jso',
        'read.return_value': json_file_bytes,
        'mimetype': 'application/octet-stream',
        'stream': io.BytesIO(json_file_bytes),
    }
    json_file = mock.Mock(**json_file_attr)

    request.method = "POST"
    request.files = {"image_file": image_file, "json_file": json_file}

    request.get_data.return_value = None

    with pytest.raises(BadInput) as e:
        test_anno_image_input.handle_request(request, predict_image_and_json)

    assert "unexpected file" in str(e.value)


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
    request.files = {"a_different_name_used": file}
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
