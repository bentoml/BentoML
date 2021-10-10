import os


def assert_have_file_extension(directory: str, ext: str):
    assert os.path.isdir(directory), f"{directory} is not a directory"
    assert any(f.endswith(ext) for f in os.listdir(directory))
