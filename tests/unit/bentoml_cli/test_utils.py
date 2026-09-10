from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from bentoml_cli.utils import build_args_option


@click.command()
@build_args_option
def command_with_build_args() -> None:
    pass


@pytest.mark.parametrize("content", ["", "- first\n- second\n"])
def test_build_args_file_requires_mapping(tmp_path: Path, content: str) -> None:
    args_file = tmp_path / "args.yaml"
    args_file.write_text(content)

    result = CliRunner().invoke(command_with_build_args, ["--arg-file", str(args_file)])

    assert result.exit_code == 2
    assert "Argument file must contain a YAML mapping" in result.output
