import pytest
import math
import pandas as pd
import numpy as np
import json

from bentoml.adapters import DataframeInput
from bentoml.adapters.dataframe_input import (
    _check_dataframe_column_contains,
    read_dataframes_from_json_n_csv,
)
from bentoml.exceptions import BadInput

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock


def test_dataframe_request_schema():
    handler = DataframeInput(
        input_dtypes={"col1": "int", "col2": "float", "col3": "string"}
    )

    schema = handler.request_schema["application/json"]["schema"]
    assert "object" == schema["type"]
    assert 3 == len(schema["properties"])
    assert "array" == schema["properties"]["col1"]["type"]
    assert "integer" == schema["properties"]["col1"]["items"]["type"]
    assert "number" == schema["properties"]["col2"]["items"]["type"]
    assert "string" == schema["properties"]["col3"]["items"]["type"]


def test_dataframe_handle_cli(capsys, tmpdir):
    def test_func(df):
        return df["name"][0]

    handler = DataframeInput()

    json_file = tmpdir.join("test.json")
    with open(str(json_file), "w") as f:
        f.write('[{"name": "john","game": "mario","city": "sf"}]')

    test_args = ["--input={}".format(json_file)]
    handler.handle_cli(test_args, test_func)
    out, _ = capsys.readouterr()
    assert out.strip().endswith("john")


def test_dataframe_handle_aws_lambda_event():
    test_content = '[{"name": "john","game": "mario","city": "sf"}]'

    def test_func(df):
        return df["name"][0]

    handler = DataframeInput()
    event = {
        "headers": {"Content-Type": "application/json"},
        "body": test_content,
    }
    response = handler.handle_aws_lambda_event(event, test_func)
    assert response["statusCode"] == 200
    assert response["body"] == '"john"'

    handler = DataframeInput()
    event_without_content_type_header = {
        "headers": {},
        "body": test_content,
    }
    response = handler.handle_aws_lambda_event(
        event_without_content_type_header, test_func
    )
    assert response["statusCode"] == 200
    assert response["body"] == '"john"'

    with pytest.raises(BadInput):
        event_with_bad_input = {
            "headers": {},
            "body": "bad_input_content",
        }
        handler.handle_aws_lambda_event(event_with_bad_input, test_func)


def test_check_dataframe_column_contains():
    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )

    # this should pass
    _check_dataframe_column_contains({"a": "int", "b": "int", "c": "int"}, df)
    _check_dataframe_column_contains({"a": "int"}, df)
    _check_dataframe_column_contains({"a": "int", "c": "int"}, df)

    # this should raise exception
    with pytest.raises(BadInput) as e:
        _check_dataframe_column_contains({"required_column_x": "int"}, df)
    assert "Missing columns: required_column_x" in str(e.value)

    with pytest.raises(BadInput) as e:
        _check_dataframe_column_contains(
            {"a": "int", "b": "int", "d": "int", "e": "int"}, df
        )
    assert "Missing columns:" in str(e.value)
    assert "required_column:" in str(e.value)


def test_dataframe_handle_request_csv():
    def test_function(df):
        return df["name"][0]

    handler = DataframeInput()
    csv_data = 'name,game,city\njohn,mario,sf'.encode('utf-8')
    request = Mock()
    request.headers = {'output_orient': 'records', 'orient': 'records'}
    request.content_type = 'text/csv'
    request.data = csv_data

    result = handler.handle_request(request, test_function)
    assert result.get_data().decode('utf-8') == '"john"'


def test_batch_read_dataframes_from_json_n_csv():
    for df in (
        pd.DataFrame(np.random.rand(2, 3)),
        pd.DataFrame(["str1", "str2", "str3"]),  # single dim sting array
        pd.DataFrame([np.nan]),  # special values
        pd.DataFrame([math.nan]),  # special values
        pd.DataFrame([" "]),  # special values
        # pd.DataFrame([""]),  # TODO: -> NaN
    ):
        csv_str = df.to_json()
        list_str = json.dumps(df.to_numpy().tolist())
        test_datas = (
            [csv_str.encode()] * 20
            + [list_str.encode()] * 20
            + [df.to_csv().encode()] * 20
            + [df.to_csv(index=False).encode()] * 20
        )

        test_types = (
            ['application/json'] * 20
            + ['application/json'] * 20
            + ['text/csv'] * 20
            + ['text/csv'] * 20
        )

        df_merged, slices = read_dataframes_from_json_n_csv(test_datas, test_types)
        for s in slices:
            left = df_merged[s].values
            right = df.values
            if right.dtype == np.float:
                np.testing.assert_array_almost_equal(left, right)
            else:
                np.testing.assert_array_equal(left, right)
