import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from bentoml.handlers import DataframeHandler  # noqa: E402


def test_dataframe_handle_cli(capsys, tmpdir):
    test_content = """
    [
      {
        "name": "john",
        "game": "mario",
        "city": "sf"
      }
    ]
    """

    def test_func(df):
        return df[0]['name']

    handler = DataframeHandler()

    import json
    json_file = tmpdir.join('test.json')
    with open(json_file, 'w') as f:
        f.write(test_content)

    test_args = ['--input={}'.format(json_file)]
    handler.handle_cli(test_args, test_func)
    out, err = capsys.readouterr()
    assert out.strip().endswith('john')


def test_dataframe_handle_aws_lambda_event():
    test_content = """
    {
      "name": "john",
      "game": "mario",
      "city": "sf"
    }
    """

    def test_func(json_obj):
        return json_obj['name']

    handler = DataframeHandler()
    success_event_obj = {'headers': {'Content-Type': 'application/json'}, 'body': test_content}
    success_response = handler.handle_aws_lambda_event(success_event_obj, test_func)

    assert success_response['statusCode'] == 200
    assert success_response['body'] == 'john'

    error_event_obj = {'headers': {'Content-Type': 'this_will_fail'}, 'body': test_content}
    error_response = handler.handle_aws_lambda_event(error_event_obj, test_func)
    assert error_response['statusCode'] == 400
