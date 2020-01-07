import os
import tempfile
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
    run_cmd = cli.commands["<API_NAME>"]
    result = runner.invoke(
        run_cmd,
        ["predict_dataframe", bento_bundle_path, "--input", input_path, "-o", "json"],
    )

    assert result.exit_code == 0
    assert result.output.strip() == '3'
