from __future__ import annotations

from _bentoml_sdk.images import Image


def test_image_system_packages_are_shell_quoted() -> None:
    image = Image(distro="debian")

    image.system_packages("libpq-dev", "package name", "foo$(touch /tmp/pwned)")

    assert image.commands[-1] == (
        "apt-get install -q -y -o Dpkg::Options::=--force-confdef "
        "libpq-dev 'package name' 'foo$(touch /tmp/pwned)'"
    )
