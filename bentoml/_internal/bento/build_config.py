from __future__ import annotations

import os
import re
import sys
import typing as t
import logging
import subprocess
from sys import version_info
from shlex import quote
from typing import TYPE_CHECKING

import fs
import attr
import yaml
import fs.copy
from dotenv import dotenv_values  # type: ignore
from pathspec import PathSpec

from .gen import generate_dockerfile
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from ..utils import copy_file_to_fs_folder
from .docker import DistroSpec
from .docker import get_supported_spec
from .docker import SUPPORTED_CUDA_VERSIONS
from .docker import DOCKER_SUPPORTED_DISTROS
from .docker import ALLOWED_CUDA_VERSION_ARGS
from .docker import SUPPORTED_PYTHON_VERSIONS
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ..configuration import CLEAN_BENTOML_VERSION
from .build_dev_bentoml_whl import build_bentoml_editable_wheel

if TYPE_CHECKING:
    from attr import Attribute
    from fs.base import FS

logger = logging.getLogger(__name__)


# Docker defaults
DEFAULT_CUDA_VERSION = "11.6.2"
DEFAULT_DOCKER_DISTRO = "debian"

CONDA_ENV_YAML_FILE_NAME = "environment.yml"


def _convert_python_version(py_version: str | None) -> str | None:
    if py_version is None:
        return None

    if not isinstance(py_version, str):
        py_version = str(py_version)

    match = re.match(r"^(\d{1})\.(\d{,2})(?:\.\w+)?$", py_version)
    if match is None:
        raise InvalidArgument(
            f'Invalid build option: docker.python_version="{py_version}", python '
            f"version must follow standard python semver format, e.g. 3.7.10 ",
        )
    major, minor = match.groups()
    target_python_version = f"{major}.{minor}"
    if target_python_version != py_version:
        logger.warning(
            "BentoML will install the latest python%s instead of the specified version %s. To use the exact python version, use a custom docker base image. See https://docs.bentoml.org/en/latest/concepts/bento.html#custom-base-image-advanced",
            target_python_version,
            py_version,
        )
    return target_python_version


def _convert_cuda_version(
    cuda_version: t.Optional[t.Union[str, int]]
) -> t.Optional[str]:
    if cuda_version is None or cuda_version == "" or cuda_version == "None":
        return None

    if isinstance(cuda_version, int):
        cuda_version = str(cuda_version)

    if cuda_version == "default":
        return DEFAULT_CUDA_VERSION

    if cuda_version in ALLOWED_CUDA_VERSION_ARGS:
        _cuda_version = ALLOWED_CUDA_VERSION_ARGS[cuda_version]
        if _cuda_version in SUPPORTED_CUDA_VERSIONS:
            return _cuda_version

    raise BentoMLException(
        f'Unsupported cuda version: "{cuda_version}". Supported cuda versions are {list(ALLOWED_CUDA_VERSION_ARGS.keys())}'
    )


def _convert_env(
    env: str | list[str] | dict[str, str] | None
) -> dict[str, str] | dict[str, str | None] | None:
    if env is None:
        return None

    if isinstance(env, str):
        logger.debug("Reading dot env file '%s' specified in config", env)
        return dict(dotenv_values(env))

    if isinstance(env, list):
        env_dict: dict[str, str | None] = {}
        for envvar in env:
            match = re.match(r"^(\w+)=(\w+)$", envvar)
            if not match:
                raise BentoMLException(
                    "All value in `env` list must follow format ENV=VALUE"
                )
            env_key, env_value = match.groups()
            env_dict[env_key] = env_value
        return env_dict

    if isinstance(env, dict):
        # convert all dict key and values to string
        return {str(k): str(v) for k, v in env.items()}

    raise BentoMLException(
        f"`env` must be either a list, a dict, or a path to a dot environment file, got type '{type(env)}' instead."
    )


@attr.frozen
class DockerOptions:

    # For validating user defined bentofile.yaml.
    __forbid_extra_keys__ = True
    # always omit config values in case of default values got changed in future BentoML releases
    __omit_if_default__ = False

    distro: t.Optional[str] = attr.field(
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(DOCKER_SUPPORTED_DISTROS)
        ),
    )
    python_version: t.Optional[str] = attr.field(
        converter=_convert_python_version,
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(SUPPORTED_PYTHON_VERSIONS)
        ),
    )
    cuda_version: t.Optional[str] = attr.field(
        default=None,
        converter=_convert_cuda_version,
        validator=attr.validators.optional(
            attr.validators.in_(ALLOWED_CUDA_VERSION_ARGS)
        ),
    )
    env: t.Optional[t.Union[str, t.List[str], t.Dict[str, str]]] = attr.field(
        default=None,
        converter=_convert_env,
    )
    system_packages: t.Optional[t.List[str]] = None
    setup_script: t.Optional[str] = None
    base_image: t.Optional[str] = None
    dockerfile_template: t.Optional[str] = None

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
            if self.system_packages:
                logger.warning(
                    f"docker base_image {self.base_image} is used, "
                    f"'system_packages={self.system_packages}' option is ignored.",
                )

        if self.distro is not None and self.cuda_version is not None:
            supports_cuda = get_supported_spec("cuda")
            if self.distro not in supports_cuda:
                raise BentoMLException(
                    f'Distro "{self.distro}" does not support CUDA. Distros that support CUDA are: {supports_cuda}.'
                )

    def with_defaults(self) -> DockerOptions:
        # Convert from user provided options to actual build options with default values
        defaults: t.Dict[str, t.Any] = {}

        if self.base_image is None:
            if self.distro is None:
                defaults["distro"] = DEFAULT_DOCKER_DISTRO
            if self.python_version is None:
                python_version = f"{version_info.major}.{version_info.minor}"
                defaults["python_version"] = python_version

        return attr.evolve(self, **defaults)

    def write_to_bento(
        self, bento_fs: FS, build_ctx: str, conda_options: CondaOptions
    ) -> None:
        use_conda = not conda_options.is_empty()

        docker_folder = fs.path.combine("env", "docker")
        bento_fs.makedirs(docker_folder, recreate=True)
        dockerfile = fs.path.combine(docker_folder, "Dockerfile")

        with bento_fs.open(dockerfile, "w") as dockerfile:
            dockerfile.write(
                generate_dockerfile(self, build_ctx=build_ctx, use_conda=use_conda)
            )

        copy_file_to_fs_folder(
            fs.path.join(os.path.dirname(__file__), "docker", "entrypoint.sh"),
            bento_fs,
            docker_folder,
        )

        if self.setup_script:
            try:
                setup_script = resolve_user_filepath(self.setup_script, build_ctx)
            except FileNotFoundError as e:
                raise InvalidArgument(f"Invalid setup_script file: {e}")
            copy_file_to_fs_folder(
                setup_script, bento_fs, docker_folder, "setup_script"
            )


if TYPE_CHECKING:
    CondaPipType = t.Dict[t.Literal["pip"], t.List[str]]
    DependencyType = t.List[t.Union[str, CondaPipType]]
else:
    DependencyType = list


def conda_dependencies_validator(
    _: t.Any, __: Attribute[DependencyType], value: DependencyType
) -> None:
    if not isinstance(value, list):
        raise InvalidArgument(
            f"Expected 'conda.dependencies' to be a list of dependencies, got a '{type(value)}' instead."
        )
    else:
        conda_pip: list[CondaPipType] = [x for x in value if isinstance(x, dict)]
        if len(conda_pip) > 0:
            if len(conda_pip) > 1 or "pip" not in conda_pip[0]:
                raise InvalidArgument(
                    "Expected dictionary under `conda.dependencies` to ONLY have key `pip`"
                )
            pip_list: list[str] = conda_pip[0]["pip"]
            if not all(isinstance(x, str) for x in pip_list):
                not_type_string = list(
                    map(
                        lambda x: str(type(x)),
                        filter(lambda x: not isinstance(x, str), pip_list),
                    )
                )
                raise InvalidArgument(
                    f"Expected 'conda.pip' values to be strings, got {not_type_string}"
                )


@attr.frozen
class CondaOptions:

    # User shouldn't add new fields under yaml file.
    __forbid_extra_keys__ = True
    # no need to omit since BentoML has already handled the default values.
    __omit_if_default__ = False

    environment_yml: t.Optional[str] = None
    channels: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )
    dependencies: t.Optional[DependencyType] = attr.field(
        default=None, validator=attr.validators.optional(conda_dependencies_validator)
    )
    pip: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )

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

    def write_to_bento(self, bento_fs: FS, build_ctx: str) -> None:
        if self.is_empty():
            return

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
                dst_filename=CONDA_ENV_YAML_FILE_NAME,
            )
        else:
            deps_list = [] if self.dependencies is None else self.dependencies
            if self.pip is not None:
                if any(isinstance(x, dict) for x in deps_list):
                    raise BentoMLException(
                        "Cannot not have both 'conda.dependencies.pip' and 'conda.pip'"
                    )
                deps_list.append(dict(pip=self.pip))  # type: ignore

            if not deps_list:
                return

            yaml_content: dict[str, DependencyType | list[str] | None] = dict(
                dependencies=deps_list
            )
            yaml_content["channels"] = self.channels
            with bento_fs.open(
                fs.path.combine(conda_folder, CONDA_ENV_YAML_FILE_NAME), "w"
            ) as f:
                yaml.dump(yaml_content, f)

    def with_defaults(self) -> CondaOptions:
        # Convert from user provided options to actual build options with default values
        defaults: dict[str, t.Any] = {}

        # When `channels` field was left empty, apply the community maintained
        # channel `conda-forge` as default
        if (
            (self.dependencies or self.pip)
            and not self.channels
            and not self.environment_yml
        ):
            defaults["channels"] = ["conda-forge"]

        return attr.evolve(self, **defaults)

    def is_empty(self) -> bool:
        return (
            not self.dependencies
            and not self.pip
            and not self.channels
            and not self.environment_yml
        )


@attr.frozen
class PythonOptions:

    # User shouldn't add new fields under yaml file.
    __forbid_extra_keys__ = True
    # no need to omit since BentoML has already handled the default values.
    __omit_if_default__ = False

    requirements_txt: t.Optional[str] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    packages: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )
    lock_packages: t.Optional[bool] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(bool)),
    )
    index_url: t.Optional[str] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    no_index: t.Optional[bool] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(bool)),
    )
    trusted_host: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )
    find_links: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )
    extra_index_url: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )
    pip_args: t.Optional[str] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    wheels: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )

    def __attrs_post_init__(self):
        if self.requirements_txt and self.packages:
            logger.warning(
                f'Build option python: `requirements_txt="{self.requirements_txt}"` found, will ignore the option: `packages="{self.packages}"`.'
            )
        if self.no_index and (self.index_url or self.extra_index_url):
            logger.warning(
                f'Build option python: `no_index="{self.no_index}"` found, will ignore `index_url` and `extra_index_url` option when installing PyPI packages.'
            )

    def is_empty(self) -> bool:
        return not self.requirements_txt and not self.packages

    def write_to_bento(self, bento_fs: FS, build_ctx: str) -> None:
        py_folder = fs.path.join("env", "python")
        wheels_folder = fs.path.join(py_folder, "wheels")
        bento_fs.makedirs(py_folder, recreate=True)

        # Save the python version of current build environment
        with bento_fs.open(fs.path.join(py_folder, "version.txt"), "w") as f:
            f.write(f"{version_info.major}.{version_info.minor}.{version_info.micro}")

        # Build BentoML whl from local source if BENTOML_BUNDLE_LOCAL_BUILD=True
        build_bentoml_editable_wheel(bento_fs.getsyspath(wheels_folder))

        # Move over required wheel files
        if self.wheels is not None:
            bento_fs.makedirs(wheels_folder, recreate=True)
            for whl_file in self.wheels:  # pylint: disable=not-an-iterable
                whl_file = resolve_user_filepath(whl_file, build_ctx)
                copy_file_to_fs_folder(whl_file, bento_fs, wheels_folder)

        pip_compile_compat: t.List[str] = []
        if self.index_url:
            pip_compile_compat.extend(["--index-url", self.index_url])
        if self.trusted_host:
            for host in self.trusted_host:
                pip_compile_compat.extend(["--trusted-host", host])
        if self.find_links:
            for link in self.find_links:
                pip_compile_compat.extend(["--find-links", link])
        if self.extra_index_url:
            for url in self.extra_index_url:
                pip_compile_compat.extend(["--extra-index-url", url])

        # add additional pip args that does not apply to pip-compile
        pip_args: t.List[str] = []
        pip_args.extend(pip_compile_compat)
        if self.no_index:
            pip_args.append("--no-index")
        if self.pip_args:
            pip_args.extend(self.pip_args.split())

        with bento_fs.open(fs.path.combine(py_folder, "install.sh"), "w") as f:
            args = " ".join(map(quote, pip_args)) if pip_args else ""
            install_script_content = (
                """\
#!/usr/bin/env bash
set -ex

# Parent directory https://stackoverflow.com/a/246128/8643197
BASEDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"

PIP_ARGS=(--no-warn-script-location """
                + args
                + """)

# BentoML by default generates two requirement files:
#  - ./env/python/requirements.lock.txt: all dependencies locked to its version presented during `build`
#  - ./env/python/requirements.txt: all dependencies as user specified in code or requirements.txt file
REQUIREMENTS_TXT="$BASEDIR/requirements.txt"
REQUIREMENTS_LOCK="$BASEDIR/requirements.lock.txt"
WHEELS_DIR="$BASEDIR/wheels"
BENTOML_VERSION=${BENTOML_VERSION:-"""
                + CLEAN_BENTOML_VERSION
                + """}
# Install python packages, prefer installing the requirements.lock.txt file if it exist
if [ -f "$REQUIREMENTS_LOCK" ]; then
    echo "Installing pip packages from 'requirements.lock.txt'.."
    pip install -r "$REQUIREMENTS_LOCK" "${PIP_ARGS[@]}"
else
    if [ -f "$REQUIREMENTS_TXT" ]; then
        echo "Installing pip packages from 'requirements.txt'.."
        pip install -r "$REQUIREMENTS_TXT" "${PIP_ARGS[@]}"
    fi
fi

# Install user-provided wheels
if [ -d "$WHEELS_DIR" ]; then
    echo "Installing wheels packaged in Bento.."
    pip install "$WHEELS_DIR"/*.whl "${PIP_ARGS[@]}"
fi

# Install the BentoML from PyPI if it's not already installed
if python -c "import bentoml" &> /dev/null; then
    existing_bentoml_version=$(python -c "import bentoml; print(bentoml.__version__)")
    if [ "$existing_bentoml_version" != "$BENTOML_VERSION" ]; then
        echo "WARNING: using BentoML version ${existing_bentoml_version}"
    fi
else
    pip install bentoml==$BENTOML_VERSION
fi
                    """
            )
            f.write(install_script_content)

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

        if self.lock_packages and not self.is_empty():
            # Note: "--allow-unsafe" is required for including setuptools in the
            # generated requirements.lock.txt file, and setuptool is required by
            # pyfilesystem2. Once pyfilesystem2 drop setuptools as dependency, we can
            # remove the "--allow-unsafe" flag here.

            # Note: "--generate-hashes" is purposefully not used here because it will
            # break if user includes PyPI package from version control system

            pip_compile_in = bento_fs.getsyspath(
                fs.path.combine(py_folder, "requirements.txt")
            )
            pip_compile_out = bento_fs.getsyspath(
                fs.path.combine(py_folder, "requirements.lock.txt")
            )
            pip_compile_args = [pip_compile_in]
            pip_compile_args.extend(pip_compile_compat)
            pip_compile_args.extend(
                [
                    "--quiet",
                    "--allow-unsafe",
                    "--no-header",
                    f"--output-file={pip_compile_out}",
                ]
            )
            logger.info("Locking PyPI package versions..")
            cmd = [sys.executable, "-m", "piptools", "compile"]
            cmd.extend(pip_compile_args)
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed locking PyPI packages: {e}")
                logger.error(
                    "Falling back to using user-provided package requirement specifier, equivalent to `lock_packages=False`"
                )

    def with_defaults(self) -> PythonOptions:
        # Convert from user provided options to actual build options with default values
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


if TYPE_CHECKING:
    OptionsCls = t.Union[DockerOptions, CondaOptions, PythonOptions]


def dict_options_converter(
    options_type: t.Type[OptionsCls],
) -> t.Callable[[t.Union[OptionsCls, t.Dict[str, t.Any]]], t.Any]:
    def _converter(value: t.Union[OptionsCls, t.Dict[str, t.Any]]) -> options_type:
        if isinstance(value, dict):
            return options_type(**value)
        return value

    return _converter


@attr.frozen
class BentoBuildConfig:
    """This class is intended for modeling the bentofile.yaml file where user will
    provide all the options for building a Bento. All optional build options should be
    default to None so it knows which fields are NOT SET by the user provided config,
    which makes it possible to omitted unset fields when writing a BentoBuildOptions to
    a yaml file for future use. This also applies to nested options such as the
    DockerOptions class and the PythonOptions class.
    """

    # User shouldn't add new fields under yaml file.
    __forbid_extra_keys__ = True
    # no need to omit since BentoML has already handled the default values.
    __omit_if_default__ = False

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
        use_conda = not self.conda.is_empty()
        use_cuda = self.docker.cuda_version is not None

        if use_cuda and use_conda:
            raise BentoMLException(
                "BentoML does not support using both conda dependencies and setting a CUDA version for GPU. If you need both conda and CUDA, use a custom base image or create a dockerfile_template, see https://docs.bentoml.org/en/latest/concepts/bento.html#custom-base-image-advanced"
            )

        if self.docker.distro is not None:
            if use_conda and self.docker.distro not in get_supported_spec("miniconda"):
                raise BentoMLException(
                    f"{self.docker.distro} does not supports conda. BentoML will only support conda with the following distros: {get_supported_spec('miniconda')}."
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

    def with_defaults(self) -> FilledBentoBuildConfig:
        """
        Convert from user provided options to actual build options with defaults
        values filled in.

        Returns:
            BentoBuildConfig: a new copy of self, with default values filled
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
    def from_yaml(cls, stream: t.TextIO) -> BentoBuildConfig:
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise

        try:
            return bentoml_cattr.structure(yaml_content, cls)
        except TypeError as e:
            if "missing 1 required positional argument: 'service'" in str(e):
                raise InvalidArgument(
                    'Missing required build config field "service", which indicates import path of target bentoml.Service instance. e.g.: "service: fraud_detector.py:svc"'
                ) from e
            else:
                raise InvalidArgument(str(e)) from e

    def to_yaml(self, stream: t.TextIO) -> None:
        # TODO: Save BentoBuildOptions to a yaml file
        # This is reserved for building interactive build file creation CLI
        raise NotImplementedError


@attr.define(frozen=True)
class BentoPathSpec:
    _include: PathSpec = attr.field(
        converter=lambda x: PathSpec.from_lines("gitwildmatch", x)
    )
    _exclude: PathSpec = attr.field(
        converter=lambda x: PathSpec.from_lines("gitwildmatch", x)
    )
    # we want to ignore .git folder in cases the .git folder is very large.
    git: PathSpec = attr.field(
        default=PathSpec.from_lines("gitwildmatch", [".git"]), init=False
    )

    def includes(
        self,
        path: str,
        *,
        recurse_exclude_spec: t.Optional[t.Iterable[t.Tuple[str, PathSpec]]] = None,
    ) -> bool:
        # Determine whether a path is included or not.
        # recurse_exclude_spec is a list of (path, spec) pairs.
        to_include = (
            self._include.match_file(path)
            and not self._exclude.match_file(path)
            and not self.git.match_file(path)
        )
        if to_include:
            if recurse_exclude_spec is not None:
                return not any(
                    ignore_spec.match_file(fs.path.relativefrom(ignore_parent, path))
                    for ignore_parent, ignore_spec in recurse_exclude_spec
                )
        return False

    def from_path(self, path: str) -> t.Generator[t.Tuple[str, PathSpec], None, None]:
        """
        yield (parent, exclude_spec) from .bentoignore file of a given path
        """
        fs_ = fs.open_fs(path)
        for file in fs_.walk.files(filter=[".bentoignore"]):
            dir_path = "".join(fs.path.parts(file)[:-1])
            yield dir_path, PathSpec.from_lines("gitwildmatch", fs_.open(file))


class FilledBentoBuildConfig(BentoBuildConfig):
    service: str
    description: t.Optional[str]
    labels: t.Dict[str, t.Any]
    include: t.List[str]
    exclude: t.List[str]
    docker: DockerOptions
    python: PythonOptions
    conda: CondaOptions
