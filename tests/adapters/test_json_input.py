# pylint: disable=redefined-outer-name

import contextlib
import gzip
import json
from typing import List

import pytest

from bentoml.adapters import JsonInput
from bentoml.types import HTTPRequest


@pytest.fixture()
def input_adapter():
    return JsonInput()


@pytest.fixture()
def raw_jsons(json_files) -> List[bytes]:
    with contextlib.ExitStack() as stack:
        raws = [stack.enter_context(open(f, "rb")).read() for f in json_files]
    return raws


@pytest.fixture()
def tasks(input_adapter, json_files):
    cli_args = ["--input-file"] + json_files
    return tuple(t for t in input_adapter.from_cli(cli_args))


@pytest.fixture()
def invalid_tasks(input_adapter, bin_file, unicode_file):
    cli_args = ["--input-file", bin_file, unicode_file]
    return tuple(t for t in input_adapter.from_cli(cli_args))


def test_json_from_cli(input_adapter, json_files, raw_jsons):
    cli_args = ["--input-file"] + json_files
    tasks = input_adapter.from_cli(cli_args)
    for t, b in zip(tasks, raw_jsons):
        assert t.data == b.decode()

    cli_args = ["--input"] + [r.decode() for r in raw_jsons]
    tasks = input_adapter.from_cli(cli_args)
    for t, b in zip(tasks, raw_jsons):
        assert t.data == b.decode()


def test_json_from_http(input_adapter, raw_jsons):
    requests = [HTTPRequest(body=r) for r in raw_jsons]
    requests += [
        HTTPRequest(body=gzip.compress(r), headers=(("Content-Encoding", "gzip"),))
        for r in raw_jsons
    ]
    requests += [
        HTTPRequest(body=r, headers=(("Content-Encoding", "gzip"),)) for r in raw_jsons
    ]
    tasks = map(input_adapter.from_http_request, requests)
    iter_tasks = iter(tasks)
    for b, t in zip(raw_jsons, iter_tasks):
        assert t.data == b.decode()
    for b, t in zip(raw_jsons, iter_tasks):
        assert t.data == b.decode()
    for b, t in zip(raw_jsons, iter_tasks):
        assert t.is_discarded


def test_json_from_aws_lambda_event(input_adapter, raw_jsons):
    events = [
        {"headers": {"Content-Type": "application/json"}, "body": r.decode()}
        for r in raw_jsons
    ]
    tasks = map(input_adapter.from_aws_lambda_event, events)
    for t, b in zip(tasks, raw_jsons):
        assert t.data == b.decode()

    events = [
        {"headers": {"Content-Type": "this_will_also_work"}, "body": r.decode()}
        for r in raw_jsons
    ]
    tasks = map(input_adapter.from_aws_lambda_event, events)
    for t, b in zip(tasks, raw_jsons):
        assert t.data == b.decode()

    raw_jsons = [b"not a valid json {}"]
    events = [
        {"headers": {"Content-Type": "application/json"}, "body": r.decode()}
        for r in raw_jsons
    ]
    tasks = map(input_adapter.from_aws_lambda_event, events)
    for t, b in zip(tasks, raw_jsons):
        assert t.data == b.decode()


def test_json_extract(input_adapter, tasks, invalid_tasks):
    api_args = input_adapter.extract_user_func_args(tasks + invalid_tasks)
    json_obj_list = api_args[0]
    assert len(json_obj_list) == len(tasks)

    for out, task in zip(json_obj_list, tasks):
        assert out == json.loads(task.data)

    for task in invalid_tasks:
        assert task.is_discarded
        assert task.error.err_msg
        assert task.error.http_status != 200
        assert task.error.cli_status != 0
