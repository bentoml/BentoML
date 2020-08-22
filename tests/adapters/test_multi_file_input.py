# pylint: disable=redefined-outer-name
import pytest

from bentoml.types import HTTPRequest
from bentoml.adapters import MultiFileInput


@pytest.fixture()
def input_adapter():
    return MultiFileInput(input_names=['x', 'y'])


def test_file_input_cli(input_adapter, bin_file):
    test_args = ["--input-file-x", bin_file, "--input-file-y", bin_file]
    for task in input_adapter.from_cli(test_args):
        assert b'\x810\x899' == task.data[0].read()
        assert b'\x810\x899' == task.data[1].read()


def test_file_input_cli_list(input_adapter, bin_files):
    test_args = ["--input-file-x"] + bin_files + ["--input-file-y"] + bin_files
    tasks = input_adapter.from_cli(test_args)
    for i, task in enumerate(tasks):
        assert b'\x810\x899' + str(i).encode() == task.data[0].read()
        assert b'\x810\x899' + str(i).encode() == task.data[1].read()


def test_file_input_aws_lambda_event(input_adapter, bin_file):
    file_bytes = open(str(bin_file), 'rb').read()

    headers = {"Content-Type": "multipart/form-data; boundary=123456"}
    body = (
        b'--123456\n'
        + b'Content-Disposition: form-data; name="x"; filename="text.jpg"\n'
        + b'Content-Type: application/octet-stream\n\n'
        + file_bytes
        + b'\n--123456\n'
        + b'Content-Disposition: form-data; name="y"; filename="text.jpg"\n'
        + b'Content-Type: application/octet-stream\n\n'
        + file_bytes
        + b'\n--123456--\n'
    )
    aws_lambda_event = {
        "headers": headers,
        "body": body,
    }

    for task in input_adapter.from_aws_lambda_event([aws_lambda_event]):
        assert b'\x810\x899' == task.data[0].read()
        assert b'\x810\x899' == task.data[1].read()


def test_file_input_http_request_multipart_form(input_adapter, bin_file):
    file_bytes = open(str(bin_file), 'rb').read()

    headers = (("Content-Type", "multipart/form-data; boundary=123456"),)
    body = (
        b'--123456\n'
        + b'Content-Disposition: form-data; name="x"; filename="text.jpg"\n'
        + b'Content-Type: application/octet-stream\n\n'
        + file_bytes
        + b'\n--123456\n'
        + b'Content-Disposition: form-data; name="y"; filename="text.jpg"\n'
        + b'Content-Type: application/octet-stream\n\n'
        + file_bytes
        + b'\n--123456--\n'
    )
    request = HTTPRequest(headers=headers, body=body)
    for task in input_adapter.from_http_request([request]):
        assert b'\x810\x899' == task.data[0].read()
        assert b'\x810\x899' == task.data[1].read()


def test_file_input_http_request_malformatted_input_missing_file(
    input_adapter, bin_file
):
    file_bytes = open(str(bin_file), 'rb').read()

    headers = (("Content-Type", "multipart/form-data; boundary=123456"),)
    body = b''
    request = HTTPRequest(headers=headers, body=body)
    for task in input_adapter.from_http_request([request]):
        assert task.is_discarded

    headers = (("Content-Type", "images/jpeg"),)
    body = file_bytes
    request = HTTPRequest(headers=headers, body=body)
    for task in input_adapter.from_http_request([request]):
        assert task.is_discarded

    headers = (("Content-Type", "images/jpeg"),)
    body = (
        b'--123456\n'
        + b'Content-Disposition: form-data; name="x"; filename="text.jpg"\n'
        + b'Content-Type: application/octet-stream\n\n'
        + file_bytes
        + b'\n--123456--\n'
    )
    request = HTTPRequest(headers=headers, body=body)
    for task in input_adapter.from_http_request([request]):
        assert task.is_discarded
