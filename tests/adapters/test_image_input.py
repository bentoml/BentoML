# pylint: disable=redefined-outer-name
import base64

import pytest

from bentoml.adapters import ImageInput
from bentoml.types import HTTPRequest


@pytest.fixture()
def input_adapter():
    return ImageInput(pilmode="L")


@pytest.fixture()
def img_bytes_list(img_files):
    results = []
    for path in img_files:
        with open(path, 'rb') as f:
            results.append(f.read())
    return results


@pytest.fixture()
def tasks(input_adapter, img_files):
    cli_args = ["--input-file"] + img_files
    return tuple(t for t in input_adapter.from_cli(cli_args))


@pytest.fixture()
def invalid_tasks(input_adapter, bin_files):
    cli_args = ["--input-file"] + bin_files
    return tuple(t for t in input_adapter.from_cli(cli_args))


def test_image_input_cli(input_adapter, img_files, img_bytes_list):
    test_args = ["--input-file", img_files[0]]
    for task in input_adapter.from_cli(test_args):
        assert task.data.read() == img_bytes_list[0]

    test_args = ["--input-file"] + img_files
    for task, result in zip(input_adapter.from_cli(test_args), img_bytes_list):
        assert task.data.read() == result


def test_file_input_aws_lambda_event(input_adapter, img_files, img_bytes_list):
    with open(str(img_files[0]), "rb") as image_file:
        content = image_file.read()
        try:
            image_bytes_encoded = base64.encodebytes(content)
        except AttributeError:
            image_bytes_encoded = base64.encodebytes(str(img_files[0]))

    aws_lambda_event = {
        "body": image_bytes_encoded,
        "headers": {"Content-Type": "images/png"},
    }

    task = input_adapter.from_aws_lambda_event(aws_lambda_event)
    assert task.data.read() == img_bytes_list[0]


def test_file_input_http_request_post_binary(input_adapter, img_bytes_list):
    img_bytes = img_bytes_list[0]

    # post binary
    request = HTTPRequest(body=img_bytes)
    task = input_adapter.from_http_request(request)
    assert img_bytes == task.data.read()

    # post as multipart/form-data
    headers = (("Content-Type", "multipart/form-data; boundary=123456"),)
    body = (
        b'--123456\n'
        + b'Content-Disposition: form-data; name="file"; filename="text.jpg"\n'
        + b'Content-Type: application/octet-stream\n\n'
        + img_bytes
        + b'\n--123456--\n'
    )
    request = HTTPRequest(headers=headers, body=body)
    task = input_adapter.from_http_request(request)
    assert img_bytes == task.data.read()


def test_image_input_extract(input_adapter, tasks, invalid_tasks):
    api_args = input_adapter.extract_user_func_args(tasks + invalid_tasks)
    obj_list = api_args[0]
    assert len(obj_list) == len(tasks)

    for out, task in zip(obj_list, tasks + invalid_tasks):
        if not task.is_discarded:
            assert out.shape == (10, 10)

    for task in invalid_tasks:
        assert task.is_discarded
        assert task.error.http_status == 400
        assert task.error.cli_status != 0
        assert task.error.err_msg
