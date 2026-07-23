from __future__ import annotations

from _bentoml_sdk.images import Image


def test_apt_sources_mirror_inserts_sed_command() -> None:
    mirror = "https://mirrors.tuna.tsinghua.edu.cn/debian"
    image = Image(distro="debian")
    image.apt_sources_mirror(mirror)

    # The sed command should be the last entry added to pre-pip commands
    assert image.commands[-1] == (
        f"sed -i 's|http://deb.debian.org/debian|{mirror}|g'"
        " /etc/apt/sources.list.d/debian.sources 2>/dev/null ||"
        f" sed -i 's|http://deb.debian.org/debian|{mirror}|g' /etc/apt/sources.list"
    )


def test_apt_sources_mirror_is_chainable() -> None:
    mirror = "https://mirrors.tuna.tsinghua.edu.cn/debian"
    image = Image(distro="debian")
    result = image.apt_sources_mirror(mirror)
    assert result is image


def test_image_system_packages_are_shell_quoted() -> None:
    image = Image(distro="debian")

    image.system_packages("libpq-dev", "package name", "foo$(touch /tmp/pwned)")

    assert image.commands[-1] == (
        "apt-get install -q -y -o Dpkg::Options::=--force-confdef "
        "libpq-dev 'package name' 'foo$(touch /tmp/pwned)'"
    )
