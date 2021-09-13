import logging
import re
import sys
import typing as t

from bentoml._internal.configuration import BENTOML_VERSION, is_pip_installed_bentoml
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)

SUPPORTED_PYTHON_VERSION: t.List = ["3.6", "3.7", "3.8"]
SUPPORTED_BASE_DISTROS: t.List = ["slim", "centos7", "centos8"]
SUPPORTED_GPU_DISTROS: t.List = SUPPORTED_BASE_DISTROS

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
    "bentoml/model-server:{release_type}-python{python_version}-{distro}{suffix}"
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

            bentoml/model-server:<release_type>-<python_version>-<distro>-<suffix>

            # suffix: runtime or cudnn
            # distro: formatted <distro><version> (ami2, centos8, centos7)
            # python_version: 3.7,3.8
            # release_type: devel or versioned (0.14.0)

        Example results:

        * bentoml/model-server:`devel-python3.7-slim`
        * bentoml/model-server:`0.14.0-python3.8-centos8-cudnn`
        * bentoml/model-server:`0.14.0-python3.7-ami2-runtime`


    Example::

    from bento import svc
    from bentoml.utils import builtin_docker_image

    svc.build(
        ...
        env=dict(
            docker_options={
                'base_image': builtin_docker_image('debian', '3.8', gpu=True)
            }
        )
    )
    """

    def __init__(
        self,
        distro: str,
        python_version: t.Optional[str] = None,
        gpu: bool = False,
    ) -> None:
        if is_pip_installed_bentoml():
            self._release_type = BENTOML_VERSION
            self._suffix = "runtime" if not gpu else "cudnn"
        else:
            self._release_type: str = "devel"
            self._suffix = ""

        if python_version:
            _py_ver: str = python_version
        else:
            _py_ver = ".".join(map(str, sys.version_info[:2]))
            logger.debug(
                f"No python_version is specified, using {_py_ver}. "
                f"List of supported python version: {SUPPORTED_PYTHON_VERSION}. "
                "If your python_version isn't supported make sure to specify, e.g: "
                "bentoml.docker.ImageProvider('centos7', python_version='3.6')"
            )
        if _py_ver not in SUPPORTED_PYTHON_VERSION:
            raise BentoMLException(
                f"Python {python_version} is not supported. "
                f"Supported Python versions include: {SUPPORTED_PYTHON_VERSION}"
            )

        self._python_version: str = _py_ver

        if "ubuntu" in distro or "debian" in distro:
            logger.debug("Using slim tags for debian based distros")
            _distro = "slim"
        elif "amazon" in distro:
            logger.debug("Convert amazonlinux tags to ami2")
            _distro = "ami2"
        else:
            _distro = distro

        if gpu and _distro not in SUPPORTED_GPU_DISTROS:
            raise BentoMLException(
                f"{_distro} with GPU={gpu} is forbidden. "
                f"GPU-supported distros: {SUPPORTED_GPU_DISTROS} "
            )

        if (
            _distro not in SUPPORTED_RELEASES_COMBINATION["devel"]
            and self._release_type == "devel"
        ):
            raise BentoMLException(f"{distro} doesn't support devel tags.")

        if self._suffix and _distro not in SUPPORTED_RELEASES_COMBINATION[self._suffix]:
            raise BentoMLException(
                f"{distro} is not yet supported. "
                "Supported distros: "
                f"{SUPPORTED_RELEASES_COMBINATION[self._suffix]}"
            )

        self._distro: str = _distro

    def __repr__(self):
        actual_suffix = "-" + self._suffix if self._suffix else ""
        return RELEASE_FORMAT.format(
            release_type=self._release_type,
            python_version=self._python_version,
            distro=self._distro,
            suffix=actual_suffix,
        )


def builtin_docker_image(*args, **kwargs):
    return repr(ImageProvider(*args, **kwargs))
