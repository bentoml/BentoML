# type: ignore[no-untyped-def]

import os
import typing as t

import numpy as np
import psutil
import pytest


@pytest.fixture()
def img_file(tmpdir) -> str:
    import PIL.Image

    img_file_ = tmpdir.join("test_img.bmp")
    img = PIL.Image.fromarray(np.random.randint(255, size=(10, 10, 3)).astype("uint8"))
    img.save(str(img_file_))
    return str(img_file_)


@pytest.fixture()
def bin_file(tmpdir) -> str:
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)


def pytest_configure(config):  # pylint: disable=unused-argument
    import os
    import sys
    import subprocess

    cmd = f"{sys.executable} {os.path.join(os.getcwd(), 'train.py')}"
    subprocess.run(cmd, shell=True, check=True)


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    from bentoml.testing.server import host_bento

    with host_bento(
        "service:svc",
        workdir=os.getcwd(),
        config_file="bentoml_config.yml",
        docker=psutil.LINUX,
    ) as host_address:
        yield host_address
