# pylint: disable=redefined-outer-name
import pytest
import json
import glob
import contextlib
from typing import List

from bentoml.adapters import JsonInput


@pytest.fixture()
def input_adapter():
    return JsonInput()


@pytest.fixture()
def raw_jsons(json_files_dir) -> List[bytes]:
    with contextlib.ExitStack() as stack:
        file_names = glob.glob(json_files_dir)
        raws = [stack.enter_context(open(f, "rb")).read() for f in file_names]
    return raws


def test_json_from_cli(input_adapter, raw_jsons):
    tasks = input_adapter.from_cli(raw_jsons, [])
    for t, b in zip(tasks, raw_jsons):
        assert t.data == b


def test_json_from_aws_lambda_event(input_adapter, raw_jsons):
    events = [
        {"headers": {"Content-Type": "application/json"}, "body": r.decode(),}
        for r in raw_jsons
    ]
    tasks = input_adapter.from_aws_lambda_event(events)
    for t, r in zip(tasks, raw_jsons):
        assert t.data == r

    events = [
        {"headers": {"Content-Type": "this_will_also_work"}, "body": r.decode(),}
        for r in raw_jsons
    ]
    tasks = input_adapter.from_aws_lambda_event(events)
    for t, r in zip(tasks, raw_jsons):
        assert t.data == r

    raw_jsons = [b"not a valid json {}"]
    events = [
        {"headers": {"Content-Type": "application/json"}, "body": r.decode(),}
        for r in raw_jsons
    ]
    tasks = input_adapter.from_aws_lambda_event(events)
    for t, r in zip(tasks, raw_jsons):
        assert t.data == r


def test_json_extract(input_adapter, raw_jsons, non_utf8_bytes):
    tasks = input_adapter.from_cli(raw_jsons, [])
    args = input_adapter.extract_user_func_args(tasks)
    json_obj_list = args[0]
    for o, r in zip(json_obj_list, raw_jsons):
        assert o == json.loads(r.decode())

    tasks = input_adapter.from_cli([non_utf8_bytes, b"not a valid json {}"], [])
    args = input_adapter.extract_user_func_args(tasks)
    json_obj_list = args[0]
    assert json_obj_list == []
    for task in tasks:
        assert task.is_discarded
        assert task.context.err_msg
        assert task.context.http_status != 200
        assert task.context.cli_status != 0
