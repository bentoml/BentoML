from __future__ import annotations

from _bentoml_sdk.images import Image


def test_image_system_packages_are_shell_quoted() -> None:
    image = Image(distro="debian")

    image.system_packages("libpq-dev", "package name", "foo$(touch /tmp/pwned)")

    assert image.commands[-1] == (
        "apt-get install -q -y -o Dpkg::Options::=--force-confdef "
        "libpq-dev 'package name' 'foo$(touch /tmp/pwned)'"
    )


def test_build_config_keeps_image_state(tmp_path, monkeypatch) -> None:
    from _bentoml_sdk.images import populate_image_from_build_config
    from bentoml._internal.bento.build_config import BentoBuildConfig

    monkeypatch.chdir(tmp_path)
    tmp_path.joinpath("setup.sh").write_text("echo setup\\n")
    image = (
        Image(distro="debian").uv_lock().run_script("setup.sh").python_packages("numpy")
    )
    commands_before = list(image.commands)

    built = populate_image_from_build_config(
        image, BentoBuildConfig(service="service:Svc"), str(tmp_path)
    )

    assert built._uv_lock
    assert built._after_pip_install
    assert list(built.scripts) == list(image.scripts)
    assert built.commands == commands_before
    assert image.commands == commands_before
    built.run("echo after")
    assert built.post_commands == ["echo after"]


def test_build_config_distro_replaces_default_install_command(tmp_path) -> None:
    from _bentoml_sdk.images import populate_image_from_build_config
    from bentoml._internal.bento.build_config import BentoBuildConfig
    from bentoml._internal.container.frontend.dockerfile import CONTAINER_METADATA

    image = Image(base_image="python:3.11-alpine", distro="debian")
    config = BentoBuildConfig(service="service:Svc", docker={"distro": "alpine"})

    built = populate_image_from_build_config(image, config, str(tmp_path))

    assert built.commands == [CONTAINER_METADATA["alpine"]["default_install_command"]]
