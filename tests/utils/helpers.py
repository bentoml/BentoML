from pathlib import Path


def assert_have_file_extension(directory: str, ext: str):
    _dir = Path(directory)
    assert _dir.is_dir(), f"{directory} is not a directory"
    assert any(f.suffix == ext for f in _dir.iterdir())
