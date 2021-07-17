# pylint: disable=redefined-outer-name
import io
import os

import pytest
from urllib3.filepost import encode_multipart_formdata

from bentoml.adapters import AnnotatedImageInput
from bentoml.types import HTTPRequest, InferenceTask


@pytest.fixture()
def input_adapter():
    return AnnotatedImageInput()


def read_bin(path):
    with open(path, "rb") as f:
        return f.read()


def test_anno_image_input_extract_args(input_adapter, img_file, json_file):
    img_io = io.BytesIO(read_bin(img_file))
    img_io.name = "test.jpg"
    json_io = io.BytesIO(read_bin(json_file))

    task = InferenceTask(data=(img_io, json_io))
    args = input_adapter.extract_user_func_args([task])

    assert args[0]
    assert args[1]

    for img, json_obj in zip(*args):
        assert img.shape == (10, 10, 3)
        assert json_obj["name"] == "kaith"


def test_anno_image_input_extract_args_custom_extension(
    input_adapter, img_file, json_file
):
    img_io = io.BytesIO(read_bin(img_file))
    img_io.name = "test.custom"
    json_io = io.BytesIO(read_bin(json_file))
    task = InferenceTask(data=(img_io, json_io))

    args = input_adapter.extract_user_func_args([task])
    for _ in zip(*args):
        pass
    assert task.is_discarded

    input_adapter = AnnotatedImageInput(accept_image_formats=["custom"])
    task = InferenceTask(data=(img_io, json_io))
    args = input_adapter.extract_user_func_args([task])
    for img, json_obj in zip(*args):
        assert img.shape == (10, 10, 3)
        assert json_obj["name"] == "kaith"


def test_anno_image_input_extract_args_missing_image(input_adapter, json_file):
    json_io = io.BytesIO(read_bin(json_file))

    task = InferenceTask(data=(None, json_io))
    args = input_adapter.extract_user_func_args([task])

    assert not args[0]

    for _ in zip(*args):
        pass
    assert task.is_discarded


def test_anno_image_input_cli(input_adapter, img_file, json_file):
    test_args = ["--input-file-annotations", json_file, "--input-file-image", img_file]
    tasks = input_adapter.from_cli(test_args)

    for task in tasks:
        assert task.data[0].read() == read_bin(img_file)
        assert task.data[1].read() == read_bin(json_file)


def test_anno_image_input_cli_relative_paths(input_adapter, img_file, json_file):
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
        "--input-file-image",
        relative_image_path,
        "--input-file-annotations",
        relative_annotation_path,
    ]
    tasks = input_adapter.from_cli(test_args)

    for task in tasks:
        assert task.data[0].read() == read_bin(img_file)
        assert task.data[1].read() == read_bin(json_file)


def test_anno_image_input_aws_lambda_event(input_adapter, img_file, json_file):
    body, content_type = encode_multipart_formdata(
        dict(
            image=("test.jpg", read_bin(img_file)),
            annotations=("test.json", read_bin(json_file)),
        )
    )
    headers = {"Content-Type": content_type}
    aws_lambda_event = {"headers": headers, "body": body}
    task = input_adapter.from_aws_lambda_event(aws_lambda_event)

    assert task.data[0].read() == read_bin(img_file)
    assert task.data[1].read() == read_bin(json_file)


def test_anno_image_input_aws_lambda_event_bad_content_type(
    input_adapter, img_file, json_file
):
    body, _ = encode_multipart_formdata(
        dict(
            image=("test.jpg", read_bin(img_file)),
            annotations=("test.json", read_bin(json_file)),
        )
    )
    headers = {"Content-Type": "image/jpeg"}
    aws_lambda_event = {"body": body, "headers": headers}
    task = input_adapter.from_aws_lambda_event(aws_lambda_event)

    assert task.is_discarded


def test_anno_image_input_http_request_multipart_form(
    input_adapter, img_file, json_file
):
    body, content_type = encode_multipart_formdata(
        dict(
            image=("test.jpg", read_bin(img_file)),
            annotations=("test.json", read_bin(json_file)),
        )
    )
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)

    task = input_adapter.from_http_request(request)
    assert task.data[0].read() == read_bin(img_file)
    assert task.data[1].read() == read_bin(json_file)


def test_anno_image_input_http_request_multipart_form_image_only(
    input_adapter, img_file
):
    body, content_type = encode_multipart_formdata(
        dict(image=("test.jpg", read_bin(img_file)),)
    )
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)

    task = input_adapter.from_http_request(request)
    assert task.data[0].read() == read_bin(img_file)
    assert task.data[1] is None


def test_anno_image_input_http_request_too_many_files(
    input_adapter, img_file, json_file
):
    body, content_type = encode_multipart_formdata(
        dict(
            image=("test.jpg", read_bin(img_file)),
            image2=("test.jpg", read_bin(img_file)),
            annotations=("test.json", read_bin(json_file)),
        )
    )
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)

    task = input_adapter.from_http_request(request)
    assert task.data[0].read() == read_bin(img_file)
    assert task.data[1].read() == read_bin(json_file)


def test_anno_image_input_http_request_two_image_files(input_adapter, img_file):
    body, content_type = encode_multipart_formdata(
        dict(
            image=("test.jpg", read_bin(img_file)),
            image2=("test.jpg", read_bin(img_file)),
        )
    )
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)

    task = input_adapter.from_http_request(request)
    assert task.data[0].read() == read_bin(img_file)
    assert task.data[1] is None


def test_anno_image_input_check_config(input_adapter):
    config = input_adapter.config
    assert isinstance(config["accept_image_formats"], list) and isinstance(
        config["pilmode"], str
    )


def test_anno_image_input_check_request_schema(input_adapter):
    assert isinstance(input_adapter.request_schema, dict)


def test_anno_image_input_check_pip_deps(input_adapter):
    assert isinstance(input_adapter.pip_dependencies, list)


def test_anno_image_input_http_request_malformatted_input_wrong_input_name(
    input_adapter, img_file, json_file
):
    body, content_type = encode_multipart_formdata(
        dict(
            wrong_image=("test.jpg", read_bin(img_file)),
            wrong_annotations=("test.json", read_bin(json_file)),
        )
    )
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)

    task = input_adapter.from_http_request(request)
    assert task.is_discarded
