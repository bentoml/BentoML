import importlib.metadata

import packaging
import pytest

from bentoml._internal.utils.pkg import pkg_version_info


def test_returns_major_minor_micro_int_tuple():
    version = pkg_version_info("packaging")
    assert len(version) == 3
    assert all(isinstance(part, int) for part in version)


def test_accepts_module_or_name():
    assert pkg_version_info(packaging) == pkg_version_info("packaging")


def test_missing_package_raises():
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        pkg_version_info("bentoml_no_such_package_9999")
