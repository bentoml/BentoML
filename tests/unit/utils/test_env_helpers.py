from typing import List

import pytest

from bentoml.utils import with_pip_install_options, find_required_pypi_packages, find_local_py_modules_used

sample_packages: List[str] = ["numpy", "requests"]


@pytest.fixture
def sample_file():
    return "./tests/unit/utils/sample.py"


def test_find_pypi_packages(sample_file):
    packages = find_required_pypi_packages(sample_file, lock_versions=False)
    assert packages == sample_packages


def test_with_pip_options():
    index_url = "https://mirror.baidu.com/pypi/simple"
    find_links="https://download.pytorch.org/whl/torch_stable.html"
    with_options = with_pip_install_options(sample_packages, index_url=index_url, find_links=find_links)
    assert with_options == ["numpy --index-url=https://mirror.baidu.com/pypi/simple --find-links=https://download.pytorch.org/whl/torch_stable.html", "requests --index-url=https://mirror.baidu.com/pypi/simple --find-links=https://download.pytorch.org/whl/torch_stable.html"]


def test_find_local_modules(sample_file):
    file_pairs = find_local_py_modules_used(sample_file)
    assert file_pairs == [("./test/unit/utils/sample.py", "sample.py")]
