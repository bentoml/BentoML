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


def test_generate_containerfile_cache_mounts_use_sharing_locked(tmp_path) -> None:
    dockerfile = generate_containerfile(
        DockerOptions(
            distro="debian",
            python_version="3.11",
        ),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )

    import re

    cache_mounts = re.findall(r"--mount=type=cache[^ ]+", dockerfile)
    assert len(cache_mounts) > 0, "expected at least one cache mount"
    for mount in cache_mounts:
        assert "sharing=locked" in mount, f"cache mount missing sharing=locked: {mount}"
