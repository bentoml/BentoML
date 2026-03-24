from __future__ import annotations

from bentoml._internal.bento.build_config import CondaOptions
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.container.generate import generate_containerfile


def test_generate_containerfile_quotes_system_packages(tmp_path) -> None:
    dockerfile = generate_containerfile(
        DockerOptions(
            distro="debian",
            python_version="3.11",
            system_packages=["libpq-dev", "package name", "foo$(touch /tmp/pwned)"],
        ),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )

    assert "libpq-dev 'package name' 'foo$(touch /tmp/pwned)'" in dockerfile
