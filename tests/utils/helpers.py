from pathlib import Path

from schema import And
from schema import Use
from schema import Schema
from schema import Optional


def assert_have_file_extension(directory: str, ext: str):
    _dir = Path(directory)
    assert _dir.is_dir(), f"{directory} is not a directory"
    assert any(f.suffix == ext for f in _dir.iterdir())
