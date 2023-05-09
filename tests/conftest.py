from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def img_file(tmpdir: str) -> str:
    """Create a random image/bmp file."""
    from PIL.Image import fromarray

    img_file_ = tmpdir.join("test_img.bmp")
    img = fromarray(np.random.randint(255, size=(10, 10, 3)).astype("uint8"))
    img.save(str(img_file_))
    return str(img_file_)


@pytest.fixture()
def bin_file(tmpdir: str) -> str:
    """Create a random binary file."""
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)
