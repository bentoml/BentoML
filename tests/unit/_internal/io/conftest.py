from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.python import Metafunc


def pytest_generate_tests(metafunc: Metafunc) -> None:
    if "use_internal_bytes_contents" in metafunc.fixturenames:
        metafunc.parametrize("use_internal_bytes_contents", [True, False])


@pytest.fixture()
def img_file(tmpdir: str) -> str:
    import numpy as np
    from PIL.Image import fromarray

    img_file_ = tmpdir.join("test_img.bmp")
    img = fromarray(np.random.randint(255, size=(10, 10, 3)).astype("uint8"))
    img.save(str(img_file_))
    return str(img_file_)


@pytest.fixture()
def bin_file(tmpdir: str) -> str:
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)
