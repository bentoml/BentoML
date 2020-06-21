import os
import tempfile

import pytest
import mock
from click.testing import CliRunner
import psutil  # noqa # pylint: disable=unused-import

from bentoml.cli import create_bento_service_cli
from bentoml.cli.bento import t_unpack_jq_like_string


def generate_test_input_file():
    import uuid

    random_id = uuid.uuid4().hex
    tempdir = tempfile.mkdtemp()
    file_path = os.path.join(tempdir, random_id + ".json")

    with open(file_path, "w") as f:
        f.write('[{"col1": 1}, {"col1": 2}]')
    return file_path


def test_unpack_jq_like_string():
    test_obj = {
        "a": "b",
        "c": {
            "d": "e",
        }
    }

    assert _unpack_jq_like_string(test_obj, "a") == test_obj["a"]
    assert _unpack_jq_like_string(test_obj, "") == test_obj
    assert _unpack_jq_like_string(test_obj, "c.d") == test_obj["c"]["d"]

    # key not found should use last working key
    assert _unpack_jq_like_string(test_obj, "b") == test_obj
    assert _unpack_jq_like_string(test_obj, "c.g") == test_obj["c"]


def test_run_command_with_input_file(bento_bundle_path):
    input_path = generate_test_input_file()
    runner = CliRunner()

    cli = create_bento_service_cli()
    run_cmd = cli.commands["run"]
    result = runner.invoke(
        run_cmd,
        [
            bento_bundle_path,
            "predict_dataframe",
            "--input",
            input_path,
            "-o",
            "json",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == '3'

    result = runner.invoke(
        run_cmd,
        [
            bento_bundle_path,
            "predict_dataframe_v1",
            "--input",
            input_path,
            "-o",
            "json",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == '3'


@pytest.mark.skipif('not psutil.POSIX')
def test_gunicorn_serve_command():
    runner = CliRunner()

    cli = create_bento_service_cli()
    gunicorn_cmd = cli.commands["serve-gunicorn"]

    with mock.patch(
        'bentoml.server.gunicorn_server.GunicornBentoServer',
    ) as mocked_class:
        runner.invoke(
            gunicorn_cmd, ["/"],
        )
        instance = mocked_class.return_value
        assert instance.run.called
