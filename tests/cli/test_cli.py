import os
import tempfile

import mock
from click.testing import CliRunner

from bentoml.cli import create_bento_service_cli


def generate_test_input_file():
    import uuid

    random_id = uuid.uuid4().hex
    tempdir = tempfile.mkdtemp()
    file_path = os.path.join(tempdir, random_id + ".json")

    with open(file_path, "w") as f:
        f.write('[{"col1": 1}, {"col1": 2}]')
    return file_path


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


def test_gunicorn_serve_command(bento_bundle_path):
    runner = CliRunner()

    cli = create_bento_service_cli()
    gunicorn_cmd = cli.commands["serve-gunicorn"]

    with mock.patch(
        'bentoml.server.gunicorn_server.GunicornBentoServer',
    ) as mocked_class:
        runner.invoke(
            gunicorn_cmd, [bento_bundle_path],
        )
        instance = mocked_class.return_value
        assert instance.run.called
