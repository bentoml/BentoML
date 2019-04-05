import os
import tempfile
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from click.testing import CliRunner

from bentoml.cli import run
from tests.utils import generate_fake_model


def generate_test_input_file():
    import uuid
    random_id = uuid.uuid4().hex()
    tempdir = tempfile.mkdtemp()
    file_path = os.path.join(tempdir, random_id + '.json')

    with open(file_path, 'w') as f:
        f.write('{"age": 1}')
    return file_path


def test_run_command_with_input_file():
    saved_path = generate_fake_model()
    input_path = generate_test_input_file()
    result = CliRunner.invoke(
        run, ['--model-path', saved_path, '--api-name', 'predict', '--input', input_path])

    assert result.exit_code == 0
