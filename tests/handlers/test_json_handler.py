import pytest

from bentoml.adapters import JsonInput
from bentoml.exceptions import BadInput


def test_json_handle_cli(capsys, tmpdir):
    def test_func(obj):
        return obj[0]["name"]

    handler = JsonInput()

    json_file = tmpdir.join("test.json")
    with open(str(json_file), "w") as f:
        f.write('[{"name": "john","game": "mario","city": "sf"}]')

    test_args = ["--input={}".format(json_file)]
    handler.handle_cli(test_args, test_func)
    out, err = capsys.readouterr()
    assert out.strip().endswith("john")


def test_json_handle_aws_lambda_event():
    test_content = '[{"name": "john","game": "mario","city": "sf"}]'

    def test_func(obj):
        return obj[0]["name"]

    handler = JsonInput()
    success_event_obj = {
        "headers": {"Content-Type": "application/json"},
        "body": test_content,
    }
    success_response = handler.handle_aws_lambda_event(success_event_obj, test_func)

    assert success_response["statusCode"] == 200
    assert success_response["body"] == '"john"'

    error_event_obj = {
        "headers": {"Content-Type": "this_will_fail"},
        "body": test_content,
    }
    with pytest.raises(BadInput) as e:
        handler.handle_aws_lambda_event(error_event_obj, test_func)

    assert "Request content-type must be 'application/json" in str(e.value)
