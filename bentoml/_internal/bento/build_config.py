import os
import re
import typing as t
import logging
from sys import version_info as pyver
from typing import TYPE_CHECKING

import fs
import attr
import yaml
import fs.copy
from dotenv import dotenv_values  # type: ignore
from piptools.scripts.compile import cli as pip_compile_cli

from .gen import generate_dockerfile
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from ..utils import copy_file_to_fs_folder
from .docker import DistroSpec
from .docker import get_supported_spec
from .docker import SUPPORTED_CUDA_VERSIONS
from .docker import DOCKER_SUPPORTED_DISTROS
from .docker import SUPPORTED_PYTHON_VERSIONS
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from .build_dev_bentoml_whl import build_bentoml_editable_wheel

if pyver >= (3, 8):
    from typing import Literal
else:  # pragma: no cover
    from typing_extensions import Literal

if TYPE_CHECKING:
    from attr import Attribute
    from fs.base import FS

if pyver >= (3, 10):
    from typing import TypeAlias
else:  # pragma: no cover
    from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

PYTHON_VERSION = f"{pyver.major}.{pyver.minor}"
PYTHON_FULL_VERSION = f"{pyver.major}.{pyver.minor}.{pyver.micro}"


# Docker defaults
DEFAULT_CUDA_VERSION = "11.6.2"
DEFAULT_DOCKER_DISTRO = "debian"


def _convert_python_version(py_version: t.Optional[str]) -> t.Optional[str]:
    if py_version is None:
        return None

    if not isinstance(py_version, str):
        py_version = str(py_version)

    match = re.match(r"^(\d+).(\d+)", py_version)
    if match is None:
        raise InvalidArgument(
            f'Invalid build option: docker.python_version="{py_version}", python '
            f"version must follow standard python semver format, e.g. 3.7.10 ",
        )
    major, minor = match.groups()
    return f"{major}.{minor}"


def _convert_cuda_version(cuda_version: t.Optional[str]) -> t.Optional[str]:
    if cuda_version is None or cuda_version == "":
        return None

    if isinstance(cuda_version, int):
        cuda_version = str(cuda_version)

    if cuda_version == "default":
        cuda_version = DEFAULT_CUDA_VERSION

    match = re.match(r"^(\d+)$", cuda_version)
    if match is not None:
        cuda_version = SUPPORTED_CUDA_VERSIONS[cuda_version]

    return cuda_version


def _convert_user_envars(
    envars: t.Optional[t.Union[t.List[str], t.Dict[str, t.Any]]]
) -> t.Optional[t.Dict[str, t.Any]]:
    if envars is None:
        return None

    if isinstance(envars, list):
        return {k: v for e in envars for k, v in e.split("=")}
    else:
        return envars


def _envars_validator(
    _: t.Any,
    attribute: "Attribute[t.Union[str, t.List[str], t.Dict[str, t.Any]]]",
    value: t.Union[str, t.List[str], t.Dict[str, t.Any]],
) -> None:
    if isinstance(value, str):
        if not os.path.exists(value) and os.path.isfile(value):
            raise InvalidArgument(
                f'`env="{value}"` is not a valid path. if `env` is parsed as a string, it should be a path to an ".env" file.'
            )

    envars_rgx = re.compile(r"(^[A-Z0-9_]+)(\=)(.*\n)")

    if isinstance(value, list):
        for envar in value:
            match = envars_rgx.match(envar)
            if match is None:
                raise InvalidArgument(f"{envar} doesn't follow format ENVAR=value")

            k, _ = envar.split("=")
            if not k.isupper():
                raise InvalidArgument(
                    f'Invalid env var defined, key "{k}" should be in uppercase.'
                )
    elif isinstance(value, dict):
        if not all(k.isupper() for k in value):
            raise InvalidArgument("All envars should be in UPPERCASE.")
    else:
        raise InvalidArgument(
            f"`env` must be either a list or a dict, got type {attribute.type} instead."
        )


@attr.frozen
class DockerOptions:
    """
    .. note::

        Refers to :ref:`concepts/bento:Docker Options` section for more information.

    Args:
        distro (:code:`str`, `optional`):
            Distros to use in a 🍱 container.

            .. note::

                If :code:`None` is given, default to `debian`.

        python_version (:code:`str`, `optional`):
            Python version to use in a 🍱 container.

            .. note::

                If :code:`None` is given, default to current python version.

        cuda_version (:code:`str| Literal["default"]`, `optional`):
            CUDA version to use in a 🍱 container.

            Currently supported CUDA versions are: 11.6.2

            .. note::

                :code:`cuda_version` also accepts :code:`"default"` to use the default version set by BentoML.
                If you need supports for other CUDA versions, please open an issue at `BentoML Issue Tracker <https://github.com/bentoml/BentoML/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc>`_

        env (:code:`dict[str, Any]`, `optional`):
            A user-provided environment variable to be passed to a given bento.
            We support the following formats:

            - Passing as a YAML list of string:

            .. code-block:: yaml

                env:
                - ENVAR=value1
                - FOO=value2

            - Passing as a YAML dictionary:

            .. code-block:: yaml

                env:
                ENVAR: value1
                FOO: value2

            - Passing path to a :code:`.env` file:

            .. code-block:: yaml

                env: /path/to/.env

            .. note ::

                Make sure to follow the correct envar format: :code:`UPPERCASE=value`

        system_packages (:code:`List[str]`, `optional`):
            A user-provided system packages that can be install to a given bento based on :code:`distro` option.
            BentoML will utilize the default system package manager for the given distro to install these packages.

            .. note ::

                System packages will be installed at  **Block** :code:`SETUP_BENTO_BASE_IMAGE`

                **Block** is a component that construct the generated Dockerfile for a given bento.
                Refers to :ref:`Dockerfile generation <reference/build_config:dockerfile generation>`.

        setup_script (:code:`str`, `optional`):
            A user-provided setup script to be executed during the construction of a 🍱 container.

        base_image (:code:`str`, `optional`):
            A user-provided base image to be used for a given 🍱 container.

            .. note ::

                If a base image is provided, :code:`distro`, :code:`python_version`, :code:`cuda_version` and :code:`system_packages` options will be ignored.

            .. warning::

                Make sure to have Python installed in this given :code:`base_image`. User can also install python via a custom :code:`setup_script`.

        dockerfile_template (:code:`str`):
            A user-provided Dockerfile template to be used for a given 🍱 container.

            .. note ::

                Refers to :ref:`reference/build_config:dockerfile generation`  for more information.
    """

    distro: str = attr.field(
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(DOCKER_SUPPORTED_DISTROS)
        ),
    )
    python_version: str = attr.field(
        converter=_convert_python_version,
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(SUPPORTED_PYTHON_VERSIONS)
        ),
    )
    cuda_version: t.Union[str, Literal["default"]] = attr.field(
        default=None,
        converter=_convert_cuda_version,
        validator=attr.validators.optional(
            attr.validators.in_(SUPPORTED_CUDA_VERSIONS.values())
        ),
    )
    env: t.Dict[str, t.Any] = attr.field(
        default=None,
        converter=_convert_user_envars,
        validator=attr.validators.optional(_envars_validator),
    )
    system_packages: t.Optional[t.List[str]] = None
    setup_script: t.Optional[str] = None
    base_image: t.Optional[str] = None
    dockerfile_template: str = attr.field(
        default=None,
        validator=attr.validators.optional(
            lambda _, __, value: os.path.exists(value) and os.path.isfile(value)
        ),
    )

    def __attrs_post_init__(self):
        if self.base_image is not None:
            if self.distro is not None:
                logger.warning(
                    f"docker base_image {self.base_image} is used, 'distro={self.distro}' option is ignored.",
                )
            if self.python_version is not None:
                logger.warning(
                    f"docker base_image {self.base_image} is used, 'python={self.python_version}' option is ignored.",
                )
            if self.cuda_version is not None:
                logger.warning(
                    f"docker base_image {self.base_image} is used, 'cuda_version={self.cuda_version}' option is ignored.",
                )
            if self.system_packages is not None:
                logger.warning(
                    f"docker base_image {self.base_image} is used, 'system_packages={self.system_packages}' option is ignored.",
                )

        if self.cuda_version is not None:
            supports_cuda = get_supported_spec("cuda")
            if self.distro not in supports_cuda:
                raise BentoMLException(
                    f'Distro "{self.distro}" does not support CUDA. Distros that support CUDA are: {supports_cuda}.'
                )

    def with_defaults(self) -> "DockerOptions":
        """
        From a user-provided :code:`DockerOptions` instance, create a new instance with default values filled.
        """
        defaults: t.Dict[str, t.Any] = {}

        if self.base_image is None:
            if self.distro is None:
                defaults["distro"] = DEFAULT_DOCKER_DISTRO
            if self.python_version is None:
                if PYTHON_VERSION not in SUPPORTED_PYTHON_VERSIONS:  # pragma: no cover
                    logger.warning(
                        f"BentoML may not work well with current python version: {PYTHON_VERSION}. Supported python versions are: {', '.join(SUPPORTED_PYTHON_VERSIONS)}"
                    )
                defaults["python_version"] = PYTHON_VERSION

        if self.system_packages is None:
            defaults["system_packages"] = []
        if self.env is None:
            defaults["env"] = {}
        if self.cuda_version is None:
            defaults["cuda_version"] = None
        if self.dockerfile_template is None:
            defaults["dockerfile_template"] = None

        return attr.evolve(self, **defaults)

    def write_to_bento(
        self, bento_fs: "FS", build_ctx: str, conda_options: "CondaOptions"
    ) -> None:
        """
        Convert all user-provided options to respective file under a 🍱 directory.

        Args:
            bento_fs (:code:`fs.base.FS`):
                The filesystem representing a bento directory.
            build_ctx (:code:`str`):
                Build context is your Python project’s working directory. This is from where you start the Python interpreter during development so that your local python modules can be imported properly. Default to current directory where :code:`bentoml build` takes place.

            conda_options (:class:`CondaOptions`):
                Given :class:`CondaOptions` instance to determine whether to support conda in the 🍱 container.

        The results target will be written to :code:`env/docker` under the bento directory. The directory contains:
            - A generated :code:`Dockerfile`
            - :code:`entrypoint.sh`
            - :code:`<setup_script>.sh` [`optional`, if specified by user]
            - A wheel file of bentoml if :code:`BENTOML_BUNDLE_LOCAL_BUILD=True`
        """
        use_conda = any(
            val is not None
            for _, val in bentoml_cattr.unstructure(conda_options).items()  # type: ignore
        )
        docker_folder = fs.path.combine("env", "docker")
        bento_fs.makedirs(docker_folder, recreate=True)
        dockerfile = fs.path.combine(docker_folder, "Dockerfile")

        with bento_fs.open(dockerfile, "w") as dockerfile:
            dockerfile.write(generate_dockerfile(self, use_conda=use_conda))

        copy_file_to_fs_folder(
            fs.path.join(os.path.dirname(__file__), "docker", "entrypoint.sh"),
            bento_fs,
            docker_folder,
        )

        # If BentoML is installed in editable mode, build bentoml whl and save to Bento
        whl_dir = fs.path.combine(docker_folder, "whl")
        build_bentoml_editable_wheel(bento_fs.getsyspath(whl_dir))

        if self.setup_script:
            try:
                setup_script = resolve_user_filepath(self.setup_script, build_ctx)
            except FileNotFoundError as e:
                raise InvalidArgument(f"Invalid setup_script file: {e}")
            copy_file_to_fs_folder(
                setup_script, bento_fs, docker_folder, "setup_script"
            )


def _docker_options_structure_hook(d: t.Any, _: t.Type[DockerOptions]) -> DockerOptions:
    # Allow bentofile yaml to have either a str or list of str for these options
    if "env" in d and isinstance(d["env"], str):
        try:
            d["env"] = dotenv_values(d["env"])
        except FileNotFoundError:
            raise BentoMLException(f"Invalid env file path: {d['env']}")

    return DockerOptions(**d)


bentoml_cattr.register_structure_hook(DockerOptions, _docker_options_structure_hook)


@attr.frozen
class CondaOptions:
    """
    .. note::

        Refers to :ref:`concepts/bento:Conda Options` section for more information.

    .. note::

        :code:`conda` is not required for :code:`bentofile.yaml` to work. If both :code:`conda` and :code:`python` are provided under **bentofile.yaml**, then :code:`conda` will be used in conjunction with :code:`python`.

    Args:
        environment_yml (:code:`str`, `optional`):
            Path to a given :code:`environment.yml` file.

            .. note ::
                if :code:`environment.yml` is provided, then :code:`channels`, :code:`dependencies` and :code:`pip` will be ignored.

        channels (:code:`list[str]`, `optional`):
            List of conda channels to be used.

            - If not specified, then :code:`channels` will be set to :code:`None`.

            - If specified, BentoML will append :code:`["defaults"]` channels to the list.

        dependencies (:code:`list[str]`, `optional`):
            List of conda dependencies to be used.

            We recommend to lock this dependencies since BentoML won't automatically lock them.

            - If not specified, then :code:`dependencies` will be set to :code:`None`.

        pip (:code:`list[str]`, `optional`):
            List of pip packages to be installed via conda.

            - If not specified, then :code:`pip` will be set to :code:`None`.
    """

    environment_yml: t.Optional[str] = None
    channels: t.Optional[t.List[str]] = None
    dependencies: t.Optional[t.List[str]] = None
    pip: t.Optional[t.List[str]] = None

    def __attrs_post_init__(self):
        if self.environment_yml is not None:
            if self.channels is not None:
                logger.warning(
                    "conda environment_yml %s is used, 'channels=%s' option is ignored",
                    self.environment_yml,
                    self.channels,
                )
            if self.dependencies is not None:
                logger.warning(
                    "conda environment_yml %s is used, 'dependencies=%s' option is ignored",
                    self.environment_yml,
                    self.dependencies,
                )
            if self.pip is not None:
                logger.warning(
                    "conda environment_yml %s is used, 'pip=%s' option is ignored",
                    self.environment_yml,
                    self.pip,
                )

    def write_to_bento(self, bento_fs: "FS", build_ctx: str) -> None:
        """
        Convert all user-provided options to respective file under a bento 🍱 directory.

        The results target will be written to :code:`env/conda` under the bento directory. The directory contains:
            - If users provides :code:`environment.yml`:
                - The file will be copied to :code:`env/conda/environment.yml`.
                - Otherwise, a :code:`environment.yml` will be generated from the remaining specified fields.
        """
        conda_folder = fs.path.join("env", "conda")
        bento_fs.makedirs(conda_folder, recreate=True)

        if self.environment_yml is not None:
            environment_yml_file = resolve_user_filepath(
                self.environment_yml, build_ctx
            )
            copy_file_to_fs_folder(
                environment_yml_file,
                bento_fs,
                conda_folder,
                dst_filename="environment.yml",
            )

            return

        deps_list = [] if self.dependencies is None else self.dependencies
        if self.pip is not None:
            deps_list.append(dict(pip=self.pip))  # type: ignore

        if not deps_list:
            return

        yaml_content = dict(dependencies=deps_list)
        yaml_content["channels"] = (
            ["defaults"] if self.channels is None else self.channels
        )
        with bento_fs.open(fs.path.join(conda_folder, "environment_yml"), "w") as f:
            yaml.dump(yaml_content, f)

    def with_defaults(self) -> "CondaOptions":
        """
        From a user-provided :code:`CondaOptions` instance, create a new instance with default values filled.
        """
        defaults: t.Dict[str, t.Any] = {}

        if self.channels is not None:
            defaults["channels"] = self.channels + ["defaults"]

        return attr.evolve(self, **defaults)


@attr.frozen
class PythonOptions:
    """
    .. note::

        Refers to :ref:`concepts/bento:Python Packages` section for more information.

    Args:

        requirements_txt (:code:`str`, `optional`):
            Path to a custom :code:`requirements.txt` file.

        packages: (:code:`list[str]`, `optional`):
            List of pip packages to be installed in a given bento.

            User can specify which packages to lock or not.

            .. code-block:: yaml

                python:
                    packages:
                    - "numpy"
                    - "matplotlib==3.5.1"
                    - "package>=0.2,<0.3"
                    - "torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu"
                    - "git+https://github.com/bentoml/bentoml.git@main"

        lock_packages: (:code:`bool`, `optional`):
            Whether to lock the packages or not.

            By default, BentoML will lock the packages for you using `pip-tools <https://github.com/jazzband/pip-tools>`_

            if :code:`lock_packages` is set to :code:`False`, then :code:`packages` won't be locked.

        index_url (:code:`str`, `optional`):
            Custom pip index URL. This is passed through :code:`--index-url` option of pip.

        no_index (:code:`bool`, `optional`):
            Whether to use :code:`--no-index` option of pip.

        trusted_host (:code:`list[str]`, `optional`):
            List of trusted hosts. This is passed through :code:`--trusted-host` option of pip.

        find_links (:code:`list[str]`, `optional`):
            List of find links. This is passed through :code:`--find-links` option of pip.

        extra_index_url (:code:`list[str]`, `optional`):
            List of extra index URLs. This is passed through :code:`--extra-index-url` option of pip.

        pip_args (:code:`str`, `optional`):
            Additional pip arguments. This is passed to :code:`pip` command when installing a package.

        wheels (:code:`list[str]`, `optional`):
            List of wheels to be included in the 🍱.

            .. note ::

                If the wheel is hosted on a local network without TLS, you can indicate that the domain is safe to pip with the :code:`trusted_host` field.
    """

    requirements_txt: t.Optional[str] = None
    packages: t.Optional[t.List[str]] = None
    lock_packages: t.Optional[bool] = None
    index_url: t.Optional[str] = None
    no_index: t.Optional[bool] = None
    trusted_host: t.Optional[t.List[str]] = None
    find_links: t.Optional[t.List[str]] = None
    extra_index_url: t.Optional[t.List[str]] = None
    pip_args: t.Optional[str] = None
    wheels: t.Optional[t.List[str]] = None

    def __attrs_post_init__(self):
        if self.requirements_txt and self.packages:
            logger.warning(
                f'Build option python: requirements_txt="{self.requirements_txt}" found, this will ignore the option: packages="{self.packages}"'
            )
        if self.no_index and (self.index_url or self.extra_index_url):
            logger.warning(
                "Build option python.no_index=True found, this will ignore index_url and extra_index_url option when installing PyPI packages"
            )

    def write_to_bento(self, bento_fs: "FS", build_ctx: str) -> None:
        """
        Convert all user-provided options to the respective files under a Bento directory.

        The results target will be written to :code:`env/python` under the bento directory. The directory contains:
            - :code:`version.txt` which contains the python version in full format: :code:`3.7.3`, :code:`3.9.0`, etc.
            - :code:`wheels/` directory which contains the wheels specified in :code:`wheels` field.
            - If users specify :code:`requirements.txt` file:
                - The :code:`requirements.txt` will be copied to the bento 🍱 directory.
                - Otherwise, a generated :code:`requirements.txt` will be included containing the pip packages specified in :code:`packages` field.
            - If :code:`lock_packages` is set to :code:`True`, then a :code:`requirements.txt.lock` file will be included.
            - :code:`pip_args.txt` file which contains the pip arguments specified in :code:`pip_args` and all of the remaining field.
        """
        py_folder = fs.path.join("env", "python")
        wheels_folder = fs.path.join(py_folder, "wheels")
        bento_fs.makedirs(py_folder, recreate=True)

        # Save the python version of current build environment
        with bento_fs.open(fs.path.join(py_folder, "version.txt"), "w") as f:
            f.write(PYTHON_FULL_VERSION)

        # Move over required wheel files
        # Note: although wheel files outside of build_ctx will also work, we should
        # discourage users from doing that
        if self.wheels is not None:
            for whl_file in self.wheels:  # pylint: disable=not-an-iterable
                whl_file = resolve_user_filepath(whl_file, build_ctx)
                copy_file_to_fs_folder(whl_file, bento_fs, wheels_folder)

        if self.requirements_txt is not None:
            requirements_txt_file = resolve_user_filepath(
                self.requirements_txt, build_ctx
            )
            copy_file_to_fs_folder(
                requirements_txt_file,
                bento_fs,
                py_folder,
                dst_filename="requirements.txt",
            )
        elif self.packages is not None:
            with bento_fs.open(fs.path.join(py_folder, "requirements.txt"), "w") as f:
                f.write("\n".join(self.packages))
        else:
            # Return early if no python packages were specified
            return

        pip_args: t.List[str] = []
        if self.no_index:
            pip_args.append("--no-index")
        if self.index_url:
            pip_args.append(f"--index-url={self.index_url}")
        if self.trusted_host:
            for item in self.trusted_host:  # pylint: disable=not-an-iterable
                pip_args.append(f"--trusted-host={item}")
        if self.find_links:
            for item in self.find_links:  # pylint: disable=not-an-iterable
                pip_args.append(f"--find-links={item}")
        if self.extra_index_url:
            for item in self.extra_index_url:  # pylint: disable=not-an-iterable
                pip_args.append(f"--extra-index-url={item}")
        if self.pip_args:
            # Additional user provided pip_args
            pip_args.append(self.pip_args)

        # write pip install args to a text file if applicable
        if pip_args:
            with bento_fs.open(fs.path.join(py_folder, "pip_args.txt"), "w") as f:
                f.write(" ".join(pip_args))

        if self.lock_packages:
            # Note: "--allow-unsafe" is required for including setuptools in the
            # generated requirements.lock.txt file, and setuptool is required by
            # pyfilesystem2. Once pyfilesystem2 drop setuptools as dependency, we can
            # remove the "--allow-unsafe" flag here.

            # Note: "--generate-hashes" is purposefully not used here because it will
            # break if user includes PyPI package from version control system
            pip_compile_in = bento_fs.getsyspath(
                fs.path.join(py_folder, "requirements.txt")
            )
            pip_compile_out = bento_fs.getsyspath(
                fs.path.join(py_folder, "requirements.lock.txt")
            )
            pip_compile_args = (
                [pip_compile_in]
                + pip_args
                + [
                    "--quiet",
                    "--allow-unsafe",
                    "--no-header",
                    f"--output-file={pip_compile_out}",
                ]
            )
            logger.info("Locking PyPI package versions..")
            click_ctx = pip_compile_cli.make_context("pip-compile", pip_compile_args)
            try:
                pip_compile_cli.invoke(click_ctx)
            except Exception as e:
                logger.error(f"Failed locking PyPI packages: {e}")
                logger.error(
                    "Falling back to using user-provided package requirement specifier, equivalent to `lock_packages=False`"
                )

    def with_defaults(self) -> "PythonOptions":
        """
        From a user-provided :code:`PythonOptions` instance, create a new instance with default values filled.
        """
        defaults: t.Dict[str, t.Any] = {}

        if self.requirements_txt is None:
            if self.lock_packages is None:
                defaults["lock_packages"] = True

        return attr.evolve(self, **defaults)


def _python_options_structure_hook(d: t.Any, _: t.Type[PythonOptions]) -> PythonOptions:
    # Allow bentofile yaml to have either a str or list of str for these options
    for field in ["trusted_host", "find_links", "extra_index_url"]:
        if field in d and isinstance(d[field], str):
            d[field] = [d[field]]

    return PythonOptions(**d)


bentoml_cattr.register_structure_hook(PythonOptions, _python_options_structure_hook)


OptionsCls: TypeAlias = t.Union[DockerOptions, CondaOptions, PythonOptions]


def dict_options_converter(
    options_type: t.Type[OptionsCls],
) -> t.Callable[[t.Union[OptionsCls, t.Dict[str, t.Any]]], t.Any]:
    def _converter(value: t.Union[OptionsCls, t.Dict[str, t.Any]]) -> options_type:
        if isinstance(value, dict):
            return options_type(**value)
        return value

    return _converter


@attr.define(frozen=True, on_setattr=None)
class BentoBuildConfig:
    """
    This class is intended for modeling the :code:`bentofile.yaml` file where user will
    provide all the options for building a Bento.

    All optional build options should be default to :code:`None` so BentoML knows
    which fields are **NOT SET** by the user provided config.

    Thus, it is possible to omitted unset fields when writing a :code:`BentoBuildConfig` to
    a yaml file for future use. This also applies to nested options such as the
    :class:`DockerOptions` class and the :class:`PythonOptions` class.
    """

    service: str
    description: t.Optional[str] = None
    labels: t.Optional[t.Dict[str, t.Any]] = None
    include: t.Optional[t.List[str]] = None
    exclude: t.Optional[t.List[str]] = None
    docker: DockerOptions = attr.field(
        factory=DockerOptions,
        converter=dict_options_converter(DockerOptions),
    )
    python: PythonOptions = attr.field(
        factory=PythonOptions,
        converter=dict_options_converter(PythonOptions),
    )
    conda: CondaOptions = attr.field(
        factory=CondaOptions,
        converter=dict_options_converter(CondaOptions),
    )

    def __attrs_post_init__(self) -> None:
        use_conda = any(
            val is not None for _, val in bentoml_cattr.unstructure(self.conda).items()
        )
        use_cuda = self.docker.cuda_version is not None

        if use_cuda and use_conda:
            logger.warning(
                f'Conda will be ignored and will not be available inside bento container when "cuda={self.docker.cuda_version}" is set.'
            )

        if self.docker.distro is not None:
            if use_conda and self.docker.distro not in get_supported_spec("miniconda"):
                raise BentoMLException(
                    f"{self.docker.distro} does not supports conda. BentoML will only support conda with the following distros: {get_supported_spec('miniconda')}."
                )
            if use_cuda and self.docker.distro not in get_supported_spec("cuda"):
                raise BentoMLException(
                    f"{self.docker.distro} does not supports cuda. BentoML will only support cuda with the following distros: {get_supported_spec('cuda')}."
                )

            _spec = DistroSpec.from_distro(
                self.docker.distro, cuda=use_cuda, conda=use_conda
            )

            if _spec is not None:
                if self.docker.python_version is not None:
                    if (
                        self.docker.python_version
                        not in _spec.supported_python_versions
                    ):
                        raise BentoMLException(
                            f"{self.docker.python_version} is not supported for {self.docker.distro}. Supported python versions are: {', '.join(_spec.supported_python_versions)}."
                        )

                if self.docker.cuda_version is not None:
                    if self.docker.cuda_version != "default" and (
                        self.docker.cuda_version not in _spec.supported_cuda_versions
                    ):
                        raise BentoMLException(
                            f"{self.docker.cuda_version} is not supported for {self.docker.distro}. Supported cuda versions are: {', '.join(_spec.supported_cuda_versions)}."
                        )

    def with_defaults(self) -> "FilledBentoBuildConfig":
        """
        Returns a copy of :class:`BentoBuildConfig` with default values filled in.
        """

        return FilledBentoBuildConfig(
            self.service,
            self.description,
            {} if self.labels is None else self.labels,
            ["*"] if self.include is None else self.include,
            [] if self.exclude is None else self.exclude,
            self.docker.with_defaults(),
            self.python.with_defaults(),
            self.conda.with_defaults(),
        )

    @classmethod
    def from_yaml(cls, stream: t.TextIO) -> "BentoBuildConfig":
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise

        try:
            return bentoml_cattr.structure(yaml_content, cls)
        except KeyError as e:
            if str(e) == "'service'":
                raise InvalidArgument(
                    'Missing required build config field "service", which'
                    " indicates import path of target bentoml.Service instance."
                    ' e.g.: "service: fraud_detector.py:svc"'
                )
            else:
                raise

    def to_yaml(self, stream: t.TextIO) -> None:
        # TODO: Save BentoBuildOptions to a yaml file
        # This is reserved for building interactive build file creation CLI
        raise NotImplementedError


class FilledBentoBuildConfig(BentoBuildConfig):
    service: str
    description: t.Optional[str]
    labels: t.Dict[str, t.Any]
    include: t.List[str]
    exclude: t.List[str]
    docker: DockerOptions
    python: PythonOptions
    conda: CondaOptions
