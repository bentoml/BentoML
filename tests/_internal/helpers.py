import os


def assert_have_file_extension(dir: str, ext: str):
    assert os.path.isdir(dir), f"{dir} is not a directory"
    assert any(f.endswith(ext) for f in os.listdir(dir))
