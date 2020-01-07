import pytest
import pandas as pd
import numpy as np

from bentoml.handlers import DataframeHandler
from bentoml.handlers.dataframe_handler import _check_dataframe_column_contains
from bentoml.exceptions import BadInput

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock


def test_dataframe_request_schema():
    handler = DataframeHandler(
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

    handler = DataframeHandler()

    json_file = tmpdir.join("test.json")
    with open(str(json_file), "w") as f:
        f.write('[{"name": "john","game": "mario","city": "sf"}]')

    test_args = ["--input={}".format(json_file)]
    handler.handle_cli(test_args, test_func)
    out, err = capsys.readouterr()
    assert out.strip().endswith("john")


def test_dataframe_handle_aws_lambda_event():
    test_content = '[{"name": "john","game": "mario","city": "sf"}]'

    def test_func(df):
        return df["name"][0]

    handler = DataframeHandler()
    success_event_obj = {
        "headers": {"Content-Type": "application/json"},
        "body": test_content,
    }
    success_response = handler.handle_aws_lambda_event(success_event_obj, test_func)

    assert success_response["statusCode"] == 200
    assert success_response["body"] == '"john"'

    with pytest.raises(BadInput):
        error_event_obj = {
            "headers": {"Content-Type": "this_will_fail"},
            "body": test_content,
        }
        handler.handle_aws_lambda_event(error_event_obj, test_func)


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

    handler = DataframeHandler()
    csv_data = 'name,game,city\njohn,mario,sf'.encode('utf-8')
    request = Mock()
    request.headers = {'output_orient': 'records', 'orient': 'records'}
    request.content_type = 'text/csv'
    request.data = csv_data

    result = handler.handle_request(request, test_function)
    assert result.get_data().decode('utf-8') == '"john"'
