# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import typing as t

import numpy as np
import pytest

from bentoml.testing.server import run_api_server


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


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    with run_api_server(bento="service:svc", config_file="bentoml_config.yml") as host:
        yield host
