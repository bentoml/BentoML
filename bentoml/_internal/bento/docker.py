import re
import sys
import typing as t
import logging

from bentoml._version import version

from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from ..configuration import is_pypi_installed_bentoml

logger = logging.getLogger(__name__)

SUPPORTED_PYTHON_VERSION: t.List[str] = ["3.7", "3.8", "3.9"]
SUPPORTED_DEVEL_DISTROS: t.List[str] = ["debian", "ubi7", "ubi8"]
SUPPORTED_GPU_DISTROS: t.List[str] = ["debian", "ubi7", "ubi8"]
SUPPORTED_RUNTIME_DISTROS: t.List[str] = [
    "debian",
    "ubi7",
    "ubi8",
    "ami2",
    "alpine3.14",
]

SUPPORTED_RELEASES_COMBINATION: t.Dict[str, t.List[str]] = {
    "cudnn": SUPPORTED_GPU_DISTROS,
    "devel": SUPPORTED_DEVEL_DISTROS,
    "runtime": SUPPORTED_RUNTIME_DISTROS,
}

SEMVER_REGEX = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)?$")

RELEASE_FORMAT = (
    "bentoml/bento-server:{bentoml_version}-python{python_version}-{distro}{suffix}"
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
            # distro: formatted <distro><version> (ami2, ubi8, ubi7)
            # python_version: 3.7,3.8
            # release_type: devel or versioned (0.14.0)

        Example results:

        * bentoml/bento-server:`devel-python3.7-debian`
        * bentoml/bento-server:`0.14.0-python3.8-ubi8-cudnn`
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
        distro: str,
        python_version: t.Optional[str] = None,
        gpu: t.Optional[bool] = None,
        devel: t.Optional[bool] = None,
    ) -> None:
        if gpu is None:
            gpu = False

        if devel:
            # Use `devel` nightly built base image
            if gpu:
                raise BentoMLException("`devel` base image does not support GPU")
            if distro not in SUPPORTED_DEVEL_DISTROS:
                raise BentoMLException(
                    f"`devel` base image does not support distro=`{distro}`"
                )

            self._release_type: str = "devel"
        else:
            self._release_type = "cudnn" if gpu else "runtime"

        if gpu and distro not in SUPPORTED_GPU_DISTROS:
            raise BentoMLException(
                f"distro '{distro}' with GPU={gpu} is not supported. "
                f"GPU-supported distros are: {SUPPORTED_GPU_DISTROS} "
            )

        if distro not in SUPPORTED_RELEASES_COMBINATION[self._release_type]:
            raise BentoMLException(
                f"{distro} is not yet supported. "
                "Supported distros: "
                f"{SUPPORTED_RELEASES_COMBINATION[self._release_type]}"
            )

        self._distro: str = distro

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
                "e.g: \"env=dict(distro='ubi7', python_version='3.7')\""
            )

        self._python_version: str = python_version

    def __repr__(self):
        if self._release_type == "devel":
            bentoml_version = "devel"
        elif is_pypi_installed_bentoml():
            bentoml_version = BENTOML_VERSION
        else:
            # find the last official bentoml release version
            bentoml_version = ".".join(version.split(".")[0:3])

        suffix = "" if self._release_type == "devel" else "-" + self._release_type
        return RELEASE_FORMAT.format(
            bentoml_version=bentoml_version,
            python_version=self._python_version,
            distro=self._distro,
            suffix=suffix,
        )
