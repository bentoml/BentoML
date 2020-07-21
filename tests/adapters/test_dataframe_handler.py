# pylint: disable=redefined-outer-name
import itertools
import time
import pytest
import math

import flask
import pandas as pd
import numpy as np
import json
import psutil  # noqa # pylint: disable=unused-import

from bentoml.utils.dataframe_util import _csv_split, _guess_orient
from bentoml.adapters import DataframeInput
from bentoml.adapters.dataframe_input import (
    check_dataframe_column_contains,
    read_dataframes_from_json_n_csv,
)
from bentoml.exceptions import BadInput

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


def test_dataframe_request_schema():
    input_adapter = DataframeInput(
        input_dtypes={"col1": "int", "col2": "float", "col3": "string"}
    )

    schema = input_adapter.request_schema["application/json"]["schema"]
    assert "object" == schema["type"]
    assert 3 == len(schema["properties"])
    assert "array" == schema["properties"]["col1"]["type"]
    assert "integer" == schema["properties"]["col1"]["items"]["type"]
    assert "number" == schema["properties"]["col2"]["items"]["type"]
    assert "string" == schema["properties"]["col3"]["items"]["type"]


def test_dataframe_handle_cli(capsys, tmpdir):
    def test_func(df):
        return df["name"][0]

    input_adapter = DataframeInput()

    json_file = tmpdir.join("test.json")
    with open(str(json_file), "w") as f:
        f.write('[{"name": "john","game": "mario","city": "sf"}]')

    test_args = ["--input={}".format(json_file)]
    input_adapter.handle_cli(test_args, test_func)
    out, _ = capsys.readouterr()
    assert out.strip().endswith("john")


def test_dataframe_handle_aws_lambda_event():
    test_content = '[{"name": "john","game": "mario","city": "sf"}]'

    def test_func(df):
        return df["name"][0]

    input_adapter = DataframeInput()
    event = {
        "headers": {"Content-Type": "application/json"},
        "body": test_content,
    }
    response = input_adapter.handle_aws_lambda_event(event, test_func)
    assert response["statusCode"] == 200
    assert response["body"] == '"john"'

    input_adapter = DataframeInput()
    event_without_content_type_header = {
        "headers": {},
        "body": test_content,
    }
    response = input_adapter.handle_aws_lambda_event(
        event_without_content_type_header, test_func
    )
    assert response["statusCode"] == 200
    assert response["body"] == '"john"'

    with pytest.raises(BadInput):
        event_with_bad_input = {
            "headers": {},
            "body": "bad_input_content",
        }
        input_adapter.handle_aws_lambda_event(event_with_bad_input, test_func)


def test_check_dataframe_column_contains():
    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )

    # this should pass
    check_dataframe_column_contains({"a": "int", "b": "int", "c": "int"}, df)
    check_dataframe_column_contains({"a": "int"}, df)
    check_dataframe_column_contains({"a": "int", "c": "int"}, df)

    # this should raise exception
    with pytest.raises(BadInput) as e:
        check_dataframe_column_contains({"required_column_x": "int"}, df)
    assert "Missing columns: required_column_x" in str(e.value)

    with pytest.raises(BadInput) as e:
        check_dataframe_column_contains(
            {"a": "int", "b": "int", "d": "int", "e": "int"}, df
        )
    assert "Missing columns:" in str(e.value)
    assert "required_column:" in str(e.value)


def test_dataframe_handle_request_csv():
    def test_function(df):
        return df["name"][0]

    input_adapter = DataframeInput()
    csv_data = 'name,game,city\njohn,mario,sf'
    request = MagicMock(spec=flask.Request)
    request.headers = (('orient', 'records'),)
    request.content_type = 'text/csv'
    request.get_data.return_value = csv_data

    result = input_adapter.handle_request(request, test_function)
    assert result.get_data().decode('utf-8') == '"john"'


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

        test_datas.extend([df.to_json(orient=orient).encode()] * 3)
        test_types.extend(['application/json'] * 3)
        df_merged, slices = read_dataframes_from_json_n_csv(
            test_datas, test_types, orient=None
        )  # auto detect orient

    test_datas.extend([df.to_csv(index=False).encode()] * 3)
    test_types.extend(['text/csv'] * 3)

    df_merged, slices = read_dataframes_from_json_n_csv(test_datas, test_types)
    for s in slices:
        assert_df_equal(df_merged[s], df)


def test_batch_read_dataframes_from_csv_other_CRLF(df):
    csv_str = df.to_csv(index=False)
    if '\r\n' in csv_str:
        csv_str = '\n'.join(_csv_split(csv_str, '\r\n')).encode()
    else:
        csv_str = '\r\n'.join(_csv_split(csv_str, '\n')).encode()
    df_merged, _ = read_dataframes_from_json_n_csv([csv_str], ['text/csv'])
    assert_df_equal(df_merged, df)


def test_batch_read_dataframes_from_json_of_orients(df, orient):
    test_datas = [df.to_json(orient=orient).encode()] * 3
    test_types = ['application/json'] * 3
    df_merged, slices = read_dataframes_from_json_n_csv(test_datas, test_types, orient)

    df_merged, slices = read_dataframes_from_json_n_csv(test_datas, test_types, orient)
    for s in slices:
        assert_df_equal(df_merged[s], df)


def test_batch_read_dataframes_from_json_with_wrong_orients(df, orient):
    test_datas = [df.to_json(orient='table').encode()] * 3
    test_types = ['application/json'] * 3

    with pytest.raises(BadInput):
        read_dataframes_from_json_n_csv(test_datas, test_types, orient)


def test_batch_read_dataframes_from_json_in_mixed_order():
    # different column order when orient=records
    df_json = b'[{"A": 1, "B": 2, "C": 3}, {"C": 6, "A": 2, "B": 4}]'
    df_merged, slices = read_dataframes_from_json_n_csv([df_json], ['application/json'])
    for s in slices:
        assert_df_equal(df_merged[s], pd.read_json(df_json))

    # different row/column order when orient=columns
    df_json1 = b'{"A": {"1": 1, "2": 2}, "B": {"1": 2, "2": 4}, "C": {"1": 3, "2": 6}}'
    df_json2 = b'{"B": {"1": 2, "2": 4}, "A": {"1": 1, "2": 2}, "C": {"1": 3, "2": 6}}'
    df_json3 = b'{"A": {"1": 1, "2": 2}, "B": {"2": 4, "1": 2}, "C": {"1": 3, "2": 6}}'
    df_merged, slices = read_dataframes_from_json_n_csv(
        [df_json1, df_json2, df_json3], ['application/json'] * 3
    )
    for s in slices:
        assert_df_equal(
            df_merged[s][["A", "B", "C"]], pd.read_json(df_json1)[["A", "B", "C"]]
        )


def test_guess_orient(df, orient):
    json_str = df.to_json(orient=orient)
    guessed_orient = _guess_orient(json.loads(json_str))
    assert orient == guessed_orient or orient in guessed_orient


@pytest.mark.skipif('not psutil.POSIX')
def test_benchmark_load_dataframes():
    '''
    read_dataframes_from_json_n_csv should be 30x faster than pd.read_json + pd.concat
    '''
    test_count = 50

    dfs = [pd.DataFrame(np.random.rand(10, 100)) for _ in range(test_count)]
    inputs = [df.to_json().encode() for df in dfs]

    time_st = time.time()
    dfs = [pd.read_json(i) for i in inputs]
    result1 = pd.concat(dfs)
    time1 = time.time() - time_st

    time_st = time.time()
    result2, _ = read_dataframes_from_json_n_csv(
        inputs, itertools.repeat('application/json'), 'columns'
    )
    time2 = time.time() - time_st

    assert_df_equal(result1, result2)
    assert time1 / time2 > 20
