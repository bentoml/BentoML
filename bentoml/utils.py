# # APIS in utils:
# conda_environment
# ImageProvider

import logging
import re
import sys
import typing as t

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.environment.local_py_modules import find_local_py_modules_used
from ._internal.environment.pip_pkg import get_pkg_version, seek_pip_packages
from ._internal.types import PathType

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


class ImageProvider(object):
    """
    BentoML-provided docker images.

    Args:
        distros (string): distro that will be used for docker image.
        python_version (string): python version for given docker image.
        gpu (bool): Optional GPU supports.
        bentoml_version (string): Optional bentoml_version.

    Raises:
         RuntimeError: ``bentoml_version`` doesn't \
                        match semantic versioning
         RuntimeError: ``python_version`` and ``distros`` \
                        is not yet supported by BentoML
         RuntimeError: invalid combination of supported GPU distros.

    Returns:

        .. code-block:: shell

            bentoml/model-server:<release_type>-<python_version>-<distros>-<suffix>

            # suffix: runtime or cudnn
            # distros: formatted <distro><version> (ami2, centos8, centos7)
            # python_version: 3.7,3.8
            # release_type: devel or versioned (0.14.0)

        Example results:

        * bentoml/model-server:`devel-python3.7-slim`
        * bentoml/model-server:`0.14.0-python3.8-centos8-cudnn`
        * bentoml/model-server:`0.14.0-python3.7-ami2-runtime`


    Example::

    import bentoml
    from bentoml.docker import ImageProvider
    from bentoml.frameworks.pytorch import PyTorchModel
    from pandas import DataFrame

    @bentoml.env(docker_base_image=ImageProvider('debian', '3.8', gpu=True))
    @bentoml.artifacts([PyTorchModel('net')])
    class Service(bentoml.BentoService):
        @bentoml.api()
        def predict(self, df: DataFrame):
            return self.net.predict(df)
    """

    # fmt: off
    _release_fmt: t.ClassVar[str] = "bentoml/model-server:{release_type}-python{python_version}-{distros}{suffix}"  # noqa: E501 # pylint: disable=line-too-long

    _singleton: t.ClassVar[t.Optional["ImageProvider"]] = None
    # fmt: on

    @inject  # type: ignore
    def __init__(
        self,
        distros: str,
        gpu: bool = False,
        python_version: t.Optional[str] = None,
        bentoml_version: t.Optional[str] = Provide[
            BentoMLContainer.bento_bundle_deployment_version
        ],
    ) -> None:
        if bentoml_version:
            major, minor, patch = bentoml_version.split(".")
            if not SEMVER_REGEX.match(bentoml_version):
                raise ValueError(
                    f"{bentoml_version} doesn't follow semantic versioning."
                )
            if not patch:
                raise ValueError(
                    f"{bentoml_version} pass semver check but have incorrect format."
                )
            # we only support the new format with 0.14.0 forward
            if int(major) == 0 and int(minor) <= 13:
                msg = BACKWARD_COMPATIBILITY_WARNING.format(
                    classname=self.__class__.__name__,
                    bentoml_version=bentoml_version,
                )
                logger.warning(msg)
                self._release_type: str = "devel"
            else:
                self._release_type = bentoml_version

        # fmt: off
        self._suffix: str = '-' + self.get_suffix(gpu) if self._release_type != 'devel' else ''  # noqa: E501

        # fmt: on
        if python_version:
            _py_ver: str = python_version
        else:
            _py_ver = ".".join(map(str, sys.version_info[:2]))
            logger.warning(
                f"No python_version is specified, using {_py_ver}. "
                f"List of supported python version: {SUPPORTED_PYTHON_VERSION}. "
                "If your python_version isn't supported make sure to specify, e.g: "
                "bentoml.docker.ImageProvider('centos7', python_version='3.6')"
            )
        if _py_ver not in SUPPORTED_PYTHON_VERSION:
            raise RuntimeError(
                f"{python_version} is not supported. "
                f"Supported Python version: {SUPPORTED_PYTHON_VERSION}"
            )

        self._python_version: str = _py_ver

        if "ubuntu" in distros or "debian" in distros:
            logger.debug("Using slim tags for debian based distros")
            _distros = "slim"
        elif "amazon" in distros:
            logger.debug("Convert amazonlinux tags to ami2")
            _distros = "ami2"
        else:
            _distros = distros

        if gpu and _distros not in SUPPORTED_GPU_DISTROS:
            raise RuntimeError(
                f"{_distros} with GPU={gpu} is forbidden. "
                f"GPU-supported distros: {SUPPORTED_GPU_DISTROS} "
            )

        if (
            _distros not in SUPPORTED_RELEASES_COMBINATION["devel"]
            and self._release_type == "devel"
        ):
            raise RuntimeError(f"{distros} doesn't support devel tags.")

        if _distros not in SUPPORTED_RELEASES_COMBINATION[self.get_suffix(gpu)]:
            raise RuntimeError(
                f"{_distros} is not yet supported. "
                "Supported distros: "
                f"{SUPPORTED_RELEASES_COMBINATION[self.get_suffix(gpu)]}"
            )

        self._distros: str = _distros

    @staticmethod
    def get_suffix(gpu: bool) -> str:
        return "runtime" if not gpu else "cudnn"

    def __repr__(self: "ImageProvider") -> str:
        return self._release_fmt.format(
            release_type=self._release_type,
            python_version=self._python_version,
            distros=self._distros,
            suffix=self._suffix,
        )


def lock_pypi_versions(package_list: t.List[str]) -> t.List[str]:
    """
    Lock versions of pypi packages in current virtualenv

    Args:
        package_list List[str]:
            List contains package names

    Raises:
        ValueError: if one package in `package_list` is not available in current virtualenv

    Returns:

        list of lines for requirements.txt

        Example Results:

        * ['numpy==1.20.3', 'pandas==1.2.4', 'scipy==1.4.1']
    """
    pkgs_with_version = []

    for pkg in package_list:
        version = get_pkg_version(pkg)
        print(pkg, version)
        if version:
            pkg_line = f"{pkg}=={version}"
            pkgs_with_version.append(pkg_line)
        else:
            # better warning or throw an exception?
            raise ValueError(f"package {pkg} is not available in current virtualenv")

    return pkgs_with_version


def with_pip_install_options(
    package_lines: t.List[str],
    index_url: t.Optional[str] = None,
    extra_index_url: t.Optional[str] = None,
    find_links: t.Optional[str] = None,
) -> t.List[str]:
    """
    Lock versions of pypi packages in current virtualenv

    Args:
        package_lines List[str]:
            List contains items each representing one line of requirements.txt

        index_url Optional[str]:
            value of --index-url

        extra_index_url Optional[str]:
            value of --extra_index-url

        find_links Optional[str]:
            value of --find-links

    Returns:

        list of lines for requirements.txt

        Example Results:

        * ['pandas==1.2.4 --index-url=https://mirror.baidu.com/pypi/simple', 'numpy==1.20.3 --index-url=https://mirror.baidu.com/pypi/simple']
    """

    options = []
    if index_url:
        options.append(f"--index-url={index_url}")
    if extra_index_url:
        options.append(f"--extra-index-url={extra_index_url}")
    if find_links:
        options.append(f"--find-links={find_links}")

    if not options:
        return package_lines

    option_str = " ".join(options)
    pkgs_with_options = [pkg + " " + option_str for pkg in package_lines]
    return pkgs_with_options


def find_required_pypi_packages(
    path: PathType, lock_versions: bool = True
) -> t.List[str]:
    """
    Find required pypi packages in a python source file

    Args:
        path (`Union[str, bytes, os.PathLike]`):
            Path to a python source file

        lock_versions bool:
            if the versions of packages should be locked

    Returns:

        list of lines for requirements.txt

        Example Results:

        * ['numpy==1.20.3', 'pandas==1.2.4']
        * ['numpy', 'pandas']
    """
    reqs, unknown_modules = seek_pip_packages(path)
    for module_name in unknown_modules:
        logger.warning("unknown package dependency for module: %s", module_name)

    if lock_versions:
        pkg_lines = ["%s==%s" % pkg for pkg in reqs.items()]
    else:
        pkg_lines = list(reqs.keys())

    return pkg_lines
