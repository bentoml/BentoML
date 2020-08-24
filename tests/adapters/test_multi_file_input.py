# pylint: disable=redefined-outer-name
from urllib3.filepost import encode_multipart_formdata
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


def test_file_input_cli_list_missing_file(input_adapter, bin_files):
    test_args = ["--input-file-x"] + bin_files
    with pytest.raises(SystemExit):
        tasks = input_adapter.from_cli(test_args)
        for task in tasks:
            assert task.is_discarded


def test_file_input_aws_lambda_event(input_adapter, bin_file):
    file_bytes = open(str(bin_file), 'rb').read()

    body, content_type = encode_multipart_formdata(
        dict(x=("test.bin", file_bytes), y=("test.bin", file_bytes),)
    )
    headers = {"Content-Type": content_type}
    aws_lambda_event = {
        "headers": headers,
        "body": body,
    }

    for task in input_adapter.from_aws_lambda_event([aws_lambda_event]):
        assert b'\x810\x899' == task.data[0].read()
        assert b'\x810\x899' == task.data[1].read()


def test_file_input_http_request_multipart_form(input_adapter, bin_file):
    file_bytes = open(str(bin_file), 'rb').read()

    body, content_type = encode_multipart_formdata(
        dict(x=("test.bin", file_bytes), y=("test.bin", file_bytes),)
    )

    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)
    for task in input_adapter.from_http_request([request]):
        assert b'\x810\x899' == task.data[0].read()
        assert b'\x810\x899' == task.data[1].read()


def test_file_input_http_request_malformatted_input_missing_file(
    input_adapter, bin_file
):
    file_bytes = open(str(bin_file), 'rb').read()
    requests = []

    body = b''
    headers = (("Content-Type", "multipart/form-data; boundary=123456"),)
    requests.append(HTTPRequest(headers=headers, body=body))

    body = file_bytes
    headers = (("Content-Type", "images/jpeg"),)
    requests.append(HTTPRequest(headers=headers, body=body))

    body, content_type = encode_multipart_formdata(dict(x=("test.bin", file_bytes),))
    headers = (("Content-Type", content_type),)
    requests.append(HTTPRequest(headers=headers, body=body))

    for task in input_adapter.from_http_request(requests):
        assert task.is_discarded


def test_file_input_http_request_none_file(bin_file):
    file_bytes = open(str(bin_file), 'rb').read()
    allow_none_input_adapter = MultiFileInput(input_names=["x", "y"], allow_none=True)

    body, content_type = encode_multipart_formdata(dict(x=("test.bin", file_bytes),))
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)
    for task in allow_none_input_adapter.from_http_request([request]):
        assert not task.is_discarded
        assert b'\x810\x899' == task.data[0].read()
        assert task.data[1] is None
