from __future__ import annotations

from bentoml._internal.bento.build_config import CondaOptions
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.container.generate import generate_containerfile


def test_generate_containerfile_dockerfile_template_outside_cwd(tmp_path) -> None:
    # Regression test for https://github.com/bentoml/BentoML/issues/5566.
    # During `bentoml containerize` the build context is a BentoML-managed temp
    # directory (not under os.getcwd()), so resolve_user_filepath must be called
    # with secure=False — otherwise it raises ValueError for any /tmp path.
    template_dir = tmp_path / "env" / "docker"
    template_dir.mkdir(parents=True)
    (template_dir / "Dockerfile.template").write_text(
        "{% extends bento_base_template %}\n"
    )

    # Must not raise ValueError even though tmp_path is outside os.getcwd()
    generate_containerfile(
        DockerOptions(
            distro="debian",
            python_version="3.11",
            dockerfile_template="env/docker/Dockerfile.template",
        ),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )


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
