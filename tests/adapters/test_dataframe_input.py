# pylint: disable=redefined-outer-name

import itertools
import json
import math
import time

import numpy as np
import pandas as pd
import psutil
import pytest

from bentoml.adapters import DataframeInput
from bentoml.adapters.dataframe_input import read_dataframes_from_json_n_csv
from bentoml.types import HTTPRequest
from bentoml.utils.csv import csv_splitlines
from bentoml.utils.dataframe_util import guess_orient


def test_dataframe_request_schema():
    input_adapter = DataframeInput(
        dtype={"col1": "int", "col2": "float", "col3": "string"}
    )

    schema = input_adapter.request_schema["application/json"]["schema"]
    assert "object" == schema["type"]
    assert 3 == len(schema["properties"])
    assert "array" == schema["properties"]["col1"]["type"]
    assert "integer" == schema["properties"]["col1"]["items"]["type"]
    assert "number" == schema["properties"]["col2"]["items"]["type"]
    assert "string" == schema["properties"]["col3"]["items"]["type"]


def test_dataframe_handle_cli(capsys, make_api, tmpdir):
    def test_func(df):
        return df["name"]

    input_adapter = DataframeInput()
    api = make_api(input_adapter, test_func)

    json_file = tmpdir.join("test.csv")
    with open(str(json_file), "w") as f:
        f.write('name,game,city\njohn,mario,sf')

    test_args = ["--input-file", str(json_file), "--format", "csv"]
    api.handle_cli(test_args)
    out, _ = capsys.readouterr()
    assert "john" in out


def test_dataframe_handle_aws_lambda_event(make_api):
    test_content = '[{"name": "john","game": "mario","city": "sf"}]'

    def test_func(df):
        return df["name"]

    input_adapter = DataframeInput()
    api = make_api(input_adapter, test_func)
    event = {
        "headers": {"Content-Type": "application/json"},
        "body": test_content,
    }
    response = api.handle_aws_lambda_event(event)
    assert response["statusCode"] == 200
    assert response["body"] == '[{"name":"john"}]'

    event_without_content_type_header = {
        "headers": {},
        "body": test_content,
    }
    response = api.handle_aws_lambda_event(event_without_content_type_header)
    assert response["statusCode"] == 200
    assert response["body"] == '[{"name":"john"}]'

    event_with_bad_input = {
        "headers": {},
        "body": "bad_input_content",
    }
    response = api.handle_aws_lambda_event(event_with_bad_input)
    assert response["statusCode"] == 400


def test_dataframe_handle_request_csv(make_api):
    def test_func(df):
        return df["name"]

    input_adapter = DataframeInput()
    api = make_api(input_adapter, test_func)
    csv_data = b'name,game,city\njohn,mario,sf'
    request = HTTPRequest(headers={'Content-Type': 'text/csv'}, body=csv_data)
    result = api.handle_request(request)
    assert result.body == '[{"name":"john"}]'


def assert_df_equal(left: pd.DataFrame, right: pd.DataFrame):
    '''
    Compare two instances of pandas.DataFrame ignoring index and columns
    '''
    try:
        left_array = left.values
        right_array = right.values
        if right_array.dtype == np.float:
            np.testing.assert_array_almost_equal(left_array, right_array)
        else:
            np.testing.assert_array_equal(left_array, right_array)
    except AssertionError:
        raise AssertionError(
            f"\n{left.to_string()}\n is not equal to \n{right.to_string()}\n"
        )


DF_CASES = (
    pd.DataFrame(np.random.rand(1, 3)),
    pd.DataFrame(np.random.rand(2, 3)),
    pd.DataFrame(np.random.rand(2, 3), columns=['A', 'B', 'C']),
    pd.DataFrame(["str1", "str2", "str3"]),  # single dim sting array
    pd.DataFrame([np.nan]),  # special values
    pd.DataFrame([math.nan]),  # special values
    pd.DataFrame([" ", 'a"b', "a,b", "a\nb"]),  # special values
    pd.DataFrame({"test": [" ", 'a"b', "a,b", "a\nb"]}),  # special values
    # pd.Series(np.random.rand(2)),  # TODO: Series support
    # pd.DataFrame([""]),  # TODO: -> NaN
)


@pytest.fixture(params=DF_CASES)
def df(request):
    return request.param


@pytest.fixture(params=pytest.DF_ORIENTS)
def orient(request):
    return request.param


def test_batch_read_dataframes_from_mixed_json_n_csv(df):
    test_datas = []
    test_types = []

    # test content_type=application/json with various orients
    for orient in pytest.DF_ORIENTS:
        try:
            assert_df_equal(df, pd.read_json(df.to_json(orient=orient)))
        except (AssertionError, ValueError):
            # skip cases not supported by official pandas
            continue

        test_datas.extend([df.to_json(orient=orient)] * 3)
        test_types.extend(['json'] * 3)

    test_datas.extend([df.to_csv(index=False)] * 3)
    test_types.extend(['csv'] * 3)

    df_merged, counts = read_dataframes_from_json_n_csv(test_datas, test_types)
    i = 0
    for count in counts:
        assert_df_equal(df_merged[i : i + count], df)
        i += count


def test_batch_read_dataframes_from_csv_other_CRLF(df):
    csv_str = df.to_csv(index=False)

    if '\r\n' in csv_str:
        csv_str = '\n'.join(csv_splitlines(csv_str))
    else:
        csv_str = '\r\n'.join(csv_splitlines(csv_str))
    df_merged, _ = read_dataframes_from_json_n_csv([csv_str], ['csv'])
    assert_df_equal(df_merged, df)


def test_batch_read_dataframes_from_json_of_orients(df, orient):
    test_datas = [df.to_json(orient=orient)] * 3
    test_types = ['json'] * 3
    df_merged, counts = read_dataframes_from_json_n_csv(test_datas, test_types, orient)
    i = 0
    for count in counts:
        assert_df_equal(df_merged[i : i + count], df)
        i += count


def test_batch_read_dataframes_from_json_with_wrong_orients(df, orient):
    test_datas = [df.to_json(orient='table')] * 3
    test_types = ['json'] * 3

    df_merged, counts = read_dataframes_from_json_n_csv(test_datas, test_types, orient)
    assert not df_merged
    for count in counts:
        assert not count


def test_batch_read_dataframes_from_json_in_mixed_order():
    # different column order when orient=records
    df_json = '[{"A": 1, "B": 2, "C": 3}, {"C": 6, "A": 2, "B": 4}]'
    df_merged, counts = read_dataframes_from_json_n_csv([df_json], ['json'])
    i = 0
    for count in counts:
        assert_df_equal(df_merged[i : i + count], pd.read_json(df_json))
        i += count

    # different row/column order when orient=columns
    df_json1 = '{"A": {"1": 1, "2": 2}, "B": {"1": 2, "2": 4}, "C": {"1": 3, "2": 6}}'
    df_json2 = '{"B": {"1": 2, "2": 4}, "A": {"1": 1, "2": 2}, "C": {"1": 3, "2": 6}}'
    df_json3 = '{"A": {"1": 1, "2": 2}, "B": {"2": 4, "1": 2}, "C": {"1": 3, "2": 6}}'
    df_merged, counts = read_dataframes_from_json_n_csv(
        [df_json1, df_json2, df_json3], ['json'] * 3
    )
    i = 0
    for count in counts:
        assert_df_equal(
            df_merged[i : i + count][["A", "B", "C"]],
            pd.read_json(df_json1)[["A", "B", "C"]],
        )
        i += count


def test_guess_orient(df, orient):
    json_str = df.to_json(orient=orient)
    guessed_orient = guess_orient(json.loads(json_str), strict=True)
    assert orient == guessed_orient or orient in guessed_orient


@pytest.mark.skipif(not psutil.POSIX, reason="production server only works on POSIX")
def test_benchmark_load_dataframes():
    '''
    read_dataframes_from_json_n_csv should be 30x faster than pd.read_json + pd.concat
    '''
    test_count = 50

    dfs = [pd.DataFrame(np.random.rand(10, 100)) for _ in range(test_count)]
    inputs = [df.to_json() for df in dfs]

    time_st = time.time()
    dfs = [pd.read_json(i) for i in inputs]
    result1 = pd.concat(dfs)
    time1 = time.time() - time_st

    time_st = time.time()
    result2, _ = read_dataframes_from_json_n_csv(
        inputs, itertools.repeat('json'), 'columns'
    )

    time2 = time.time() - time_st
    assert_df_equal(result1, result2)

    # 5 is just an estimate on the smaller end, which should be true for most
    # development machines and Github actions CI environment, the actual ratio depends
    # on the hardware and available computing resource
    assert time1 / time2 > 5
