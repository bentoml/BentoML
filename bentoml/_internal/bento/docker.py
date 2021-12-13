import re
import sys
import typing as t
import logging

from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from ..configuration import is_pypi_installed_bentoml

logger = logging.getLogger(__name__)

SUPPORTED_PYTHON_VERSION: t.List[str] = ["3.7", "3.8", "3.9"]
SUPPORTED_BASE_DISTROS: t.List[str] = ["slim", "centos7", "centos8"]
SUPPORTED_GPU_DISTROS: t.List[str] = SUPPORTED_BASE_DISTROS

SUPPORTED_RELEASES_COMBINATION: t.Dict[str, t.List[str]] = {
    "cudnn": SUPPORTED_BASE_DISTROS,
    "devel": SUPPORTED_BASE_DISTROS,
    "runtime": SUPPORTED_BASE_DISTROS + ["ami2", "alpine3.14"],
}

SEMVER_REGEX = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)?$")

BACKWARD_COMPATIBILITY_WARNING: str = """\
Since 1.0.0, we changed the format of docker tags, thus {classname}
will only supports image tag from 1.0.0 forward, while detected bentoml_version
is {bentoml_version}.
Refers to https://hub.docker.com/r/bentoml/model-server/
if you need older version of bentoml. Using default devel image instead...
"""  # noqa: E501

RELEASE_FORMAT = (
    "bentoml/bento-server:{release_type}-python{python_version}-{distro}{suffix}"
)


class ImageProvider(object):
    """
    BentoML-provided docker images.

    Args:
        distro (string): distro that will be used for docker image.
        python_version (string): python version for given docker image.
        gpu (bool): Optional GPU supports.

    Raises:
         BentoMLException: ``python_version`` and ``distro`` \
                        is not yet supported by BentoML
         BentoMLException: invalid combination of supported GPU distros

    Returns:

        .. code-block:: shell

            bentoml/bento-server:<release_type>-<python_version>-<distro>-<suffix>

            # suffix: runtime or cudnn
            # distro: formatted <distro><version> (ami2, centos8, centos7)
            # python_version: 3.7,3.8
            # release_type: devel or versioned (0.14.0)

        Example results:

        * bentoml/bento-server:`devel-python3.7-slim`
        * bentoml/bento-server:`0.14.0-python3.8-centos8-cudnn`
        * bentoml/bento-server:`0.14.0-python3.7-ami2-runtime`


    Example::

    from bento import svc

    svc.build(
        ...
        env=dict(
            docker=dict(
                distro='debian',
                python='3.8',
                gpu=True
            )
        )
    )
    """

    def __init__(
        self,
        distro: t.Optional[
            't.Literal["slim", "amazonlinux2", "alpine", "centos7", "centos8"]'
        ] = None,
        python_version: t.Optional[str] = None,
        gpu: t.Optional[bool] = None,
    ) -> None:
        if distro is None:
            distro = "slim"
        if gpu is None:
            gpu = False

        if is_pypi_installed_bentoml():
            self._release_type = BENTOML_VERSION
            self._suffix = "cudnn" if gpu else "runtime"
        else:
            self._release_type: str = "devel"
            self._suffix = ""

        if python_version:
            assert re.match(r"^[2,3]\.[0-9]{1,2}$", python_version), (
                f"python_version '{python_version}' should be in format "
                "'{major}.{minor}', e.g. 'python_version=\"3.7\"'"
            )
        else:
            logger.debug(f"No python_version is specified, using {python_version}.")
            python_version = ".".join(map(str, sys.version_info[:2]))

        if python_version not in SUPPORTED_PYTHON_VERSION:
            raise BentoMLException(
                f"BentoML does not provide docker image for Python {python_version}."
                f"Supported Python versions are {SUPPORTED_PYTHON_VERSION}. "
                f"You may specify a different python version for BentoML to use"
                "e.g: \"env=dict(distro='centos7', python_version='3.7')\""
            )

        self._python_version: str = python_version

        if gpu and distro not in SUPPORTED_GPU_DISTROS:
            raise BentoMLException(
                f"distro '{distro}' with GPU={gpu} is not supported. "
                f"GPU-supported distros are: {SUPPORTED_GPU_DISTROS} "
            )

        if (
            distro not in SUPPORTED_RELEASES_COMBINATION["devel"]
            and self._release_type == "devel"
        ):
            raise BentoMLException(f"{distro} doesn't support devel tags.")

        if self._suffix and distro not in SUPPORTED_RELEASES_COMBINATION[self._suffix]:
            raise BentoMLException(
                f"{distro} is not yet supported. "
                "Supported distros: "
                f"{SUPPORTED_RELEASES_COMBINATION[self._suffix]}"
            )

        self._distro: str = distro

    def __repr__(self):
        actual_suffix = "-" + self._suffix if self._suffix else ""
        return RELEASE_FORMAT.format(
            release_type=self._release_type,
            python_version=self._python_version,
            distro=self._distro,
            suffix=actual_suffix,
        )
