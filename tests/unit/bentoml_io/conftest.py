from bentoml._internal.utils.pkg import pkg_version_info


def pytest_ignore_collect():
    return pkg_version_info("pydantic")[0] < 2
