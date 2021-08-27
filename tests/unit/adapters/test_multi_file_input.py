# pylint: disable=redefined-outer-name
import io

import pytest
from urllib3.filepost import encode_multipart_formdata

from bentoml.adapters import MultiFileInput
from bentoml.types import HTTPRequest, InferenceTask


@pytest.fixture()
def input_adapter():
    return MultiFileInput(input_names=["x", "y"])


def read_bin(path):
    with open(path, "rb") as f:
        return f.read()


def test_file_input_cli(input_adapter, bin_file):
    test_args = ["--input-file-x", bin_file, "--input-file-y", bin_file]
    for task in input_adapter.from_cli(test_args):
        assert b"\x810\x899" == task.data[0].read()
        assert b"\x810\x899" == task.data[1].read()


def test_file_input_cli_list(input_adapter, bin_files):
    test_args = ["--input-file-x"] + bin_files + ["--input-file-y"] + bin_files
    tasks = input_adapter.from_cli(test_args)
    for i, task in enumerate(tasks):
        assert b"\x810\x899" + str(i).encode() == task.data[0].read()
        assert b"\x810\x899" + str(i).encode() == task.data[1].read()


def test_file_input_cli_list_missing_file(input_adapter, bin_files):
    test_args = ["--input-file-x"] + bin_files
    with pytest.raises(SystemExit):
        tasks = input_adapter.from_cli(test_args)
        for task in tasks:
            assert task.is_discarded


def test_file_input_aws_lambda_event(input_adapter, bin_file):
    file_bytes = open(str(bin_file), "rb").read()

    body, content_type = encode_multipart_formdata(
        dict(
            x=("test.bin", file_bytes),
            y=("test.bin", file_bytes),
        )
    )
    headers = {"Content-Type": content_type}
    aws_lambda_event = {"headers": headers, "body": body}

    task = input_adapter.from_aws_lambda_event(aws_lambda_event)
    assert b"\x810\x899" == task.data[0].read()
    assert b"\x810\x899" == task.data[1].read()


def test_file_input_http_request_multipart_form(input_adapter, bin_file):
    file_bytes = open(str(bin_file), "rb").read()

    body, content_type = encode_multipart_formdata(
        dict(
            x=("test.bin", file_bytes),
            y=("test.bin", file_bytes),
        )
    )
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)
    task = input_adapter.from_http_request(request)
    assert b"\x810\x899" == task.data[0].read()
    assert b"\x810\x899" == task.data[1].read()


def test_file_input_http_request_malformatted_input_missing_file(
    input_adapter, bin_file
):
    file_bytes = open(str(bin_file), "rb").read()
    requests = []

    body = b""
    headers = (("Content-Type", "multipart/form-data; boundary=123456"),)
    requests.append(HTTPRequest(headers=headers, body=body))

    body = file_bytes
    headers = (("Content-Type", "images/jpeg"),)
    requests.append(HTTPRequest(headers=headers, body=body))

    body, content_type = encode_multipart_formdata(
        dict(
            x=("test.bin", file_bytes),
        )
    )
    headers = (("Content-Type", content_type),)
    requests.append(HTTPRequest(headers=headers, body=body))

    for task in map(input_adapter.from_http_request, requests):
        assert task.is_discarded


def test_file_input_http_request_none_file(bin_file):
    file_bytes = open(str(bin_file), "rb").read()
    allow_none_input_adapter = MultiFileInput(input_names=["x", "y"], allow_none=True)

    body, content_type = encode_multipart_formdata(
        dict(
            x=("test.bin", file_bytes),
        )
    )
    headers = (("Content-Type", content_type),)
    request = HTTPRequest(headers=headers, body=body)
    task = allow_none_input_adapter.from_http_request(request)
    assert not task.is_discarded
    assert b"\x810\x899" == task.data[0].read()
    assert task.data[1] is None


def test_file_input_extract(input_adapter, bin_file):
    bin_bytes = read_bin(bin_file)
    bin_ios = [tuple(io.BytesIO(bin_bytes) for _ in range(2)) for _ in range(5)]

    tasks = [InferenceTask(data=bin_io_pair) for bin_io_pair in bin_ios]
    args = input_adapter.extract_user_func_args(tasks)

    assert args[0]
    assert args[1]

    for file1, file2 in zip(*args):
        assert b"\x810\x899" == file1.read()
        assert b"\x810\x899" == file2.read()
