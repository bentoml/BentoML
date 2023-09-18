from __future__ import annotations

import os
import re
import sys
import typing as t
import logging
import subprocess
from sys import version_info
from shlex import quote

import fs
import attr
import yaml
import psutil
import fs.copy
from pathspec import PathSpec

from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from ..utils import copy_file_to_fs_folder
from ..container import generate_containerfile
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ..utils.dotenv import parse_dotenv
from ..configuration import CLEAN_BENTOML_VERSION
from ..container.generate import BENTO_PATH
from .build_dev_bentoml_whl import build_bentoml_editable_wheel
from ..container.frontend.dockerfile import DistroSpec
from ..container.frontend.dockerfile import get_supported_spec
from ..container.frontend.dockerfile import SUPPORTED_CUDA_VERSIONS
from ..container.frontend.dockerfile import ALLOWED_CUDA_VERSION_ARGS
from ..container.frontend.dockerfile import SUPPORTED_PYTHON_VERSIONS
from ..container.frontend.dockerfile import CONTAINER_SUPPORTED_DISTROS

if t.TYPE_CHECKING:
    from attr import Attribute
    from fs.base import FS

logger = logging.getLogger(__name__)


# Docker defaults
DEFAULT_CUDA_VERSION = "11.6.2"
DEFAULT_CONTAINER_DISTRO = "debian"

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
            "BentoML will install the latest 'python%s' instead of the specified 'python%s'. To use the exact python version, use a custom docker base image. See https://docs.bentoml.org/en/latest/concepts/bento.html#custom-base-image-advanced",
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
        env_path = os.path.expanduser(os.path.expandvars(env))
        if os.path.exists(env_path):
            logger.debug("Reading dot env file '%s' specified in config", env)
            with open(env_path) as f:
                return parse_dotenv(f.read())
        raise BentoMLException(f"'{env}' is not a valid '.env' file path")

    if isinstance(env, list):
        env_dict: dict[str, str | None] = {}
        for envvar in env:
            if not re.match(r"^(\w+)=(?:[\.\w\-,\/]+)$", envvar):
                raise BentoMLException(
                    "All value in `env` list must follow format ENV=VALUE"
                )
            env_key, _, env_value = envvar.partition("=")
            if os.path.isfile(env_value):
                bento_env_path = BENTO_PATH + os.path.abspath(
                    os.path.expanduser(os.path.expandvars(env_value))
                )
                logger.info(
                    "'%s' sets to '%s', which is a file. Make sure to mount this file as a persistent volume to the container when using 'run' command: 'docker run -v %s:%s ...'",
                    env_key,
                    env_value,
                    env_value,
                    bento_env_path,
                )
                env_value = bento_env_path
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
            attr.validators.in_(CONTAINER_SUPPORTED_DISTROS)
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
                    "docker base_image %s is used, 'distro=%s' option is ignored.",
                    self.base_image,
                    self.distro,
                )
            if self.python_version is not None:
                logger.warning(
                    "docker base_image %s is used, 'python=%s' option is ignored.",
                    self.base_image,
                    self.python_version,
                )
            if self.cuda_version is not None:
                logger.warning(
                    "docker base_image %s is used, 'cuda_version=%s' option is ignored.",
                    self.base_image,
                    self.cuda_version,
                )
            if self.system_packages:
                logger.warning(
                    "docker base_image %s is used, 'system_packages=%s' option is ignored.",
                    self.base_image,
                    self.system_packages,
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
                defaults["distro"] = DEFAULT_CONTAINER_DISTRO
            if self.python_version is None:
                python_version = f"{version_info.major}.{version_info.minor}"
                defaults["python_version"] = python_version

        return attr.evolve(self, **defaults)

    def write_to_bento(self, bento_fs: FS, build_ctx: str, conda: CondaOptions) -> None:
        docker_folder = fs.path.combine("env", "docker")
        bento_fs.makedirs(docker_folder, recreate=True)
        dockerfile_path = fs.path.combine(docker_folder, "Dockerfile")

        # NOTE that by default the generated Dockerfile won't have BuildKit syntax.
        # By default, BentoML containerization will use BuildKit. To opt-out specify DOCKER_BUILDKIT=0
        bento_fs.writetext(
            dockerfile_path,
            generate_containerfile(
                self, build_ctx, conda=conda, bento_fs=bento_fs, enable_buildkit=False
            ),
        )
        copy_file_to_fs_folder(
            fs.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "container",
                "frontend",
                "dockerfile",
                "entrypoint.sh",
            ),
            bento_fs,
            docker_folder,
        )

        if self.setup_script:
            try:
                setup_script = resolve_user_filepath(self.setup_script, build_ctx)
            except FileNotFoundError as e:
                raise InvalidArgument(f"Invalid setup_script file: {e}") from None
            if not os.access(setup_script, os.X_OK):
                message = f"{setup_script} is not executable."
                if not psutil.WINDOWS:
                    raise InvalidArgument(
                        f"{message} Ensure the script has a shebang line, then run 'chmod +x {setup_script}'."
                    ) from None
                raise InvalidArgument(message) from None
            copy_file_to_fs_folder(
                setup_script, bento_fs, docker_folder, "setup_script"
            )

        # If dockerfile_template is provided, then we copy it to the Bento
        # This template then can be used later to containerize.
        if self.dockerfile_template is not None:
            copy_file_to_fs_folder(
                resolve_user_filepath(self.dockerfile_template, build_ctx),
                bento_fs,
                docker_folder,
                "Dockerfile.template",
            )

    def to_dict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)


if t.TYPE_CHECKING:
    CondaPipType = dict[t.Literal["pip"], list[str]]
    DependencyType = list[str | CondaPipType]
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


if t.TYPE_CHECKING:
    ListStr: t.TypeAlias = list[str]
    CondaYamlDict = dict[str, DependencyType | list[str]]
else:
    ListStr = list


@attr.frozen
class CondaOptions:
    # User shouldn't add new fields under yaml file.
    __forbid_extra_keys__ = True
    # no need to omit since BentoML has already handled the default values.
    __omit_if_default__ = False

    environment_yml: t.Optional[str] = None
    channels: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(ListStr)),
    )
    dependencies: t.Optional[DependencyType] = attr.field(
        default=None, validator=attr.validators.optional(conda_dependencies_validator)
    )
    pip: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(ListStr)),
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
            deps_list: DependencyType = []
            if self.dependencies is not None:
                deps_list.extend(self.dependencies)
            if self.pip is not None:
                if any(isinstance(x, dict) for x in deps_list):
                    raise BentoMLException(
                        "Cannot not have both 'conda.dependencies.pip' and 'conda.pip'"
                    )
                deps_list.append({"pip": self.pip})

            if not deps_list:
                return

            yaml_content: CondaYamlDict = {"dependencies": deps_list}
            assert self.channels is not None
            yaml_content["channels"] = self.channels
            with bento_fs.open(
                fs.path.combine(conda_folder, CONDA_ENV_YAML_FILE_NAME), "w"
            ) as f:
                yaml.dump(yaml_content, f)

    def get_python_version(self, bento_fs: FS) -> str | None:
        # Get the python version from given environment.yml file

        environment_yml = bento_fs.getsyspath(
            fs.path.join(
                "env",
                "conda",
                CONDA_ENV_YAML_FILE_NAME,
            )
        )
        if os.path.exists(environment_yml):
            with open(environment_yml, "r") as f:
                for line in f:
                    match = re.search(r"(?:python=)(\d+.\d+)$", line)
                    if match:
                        return match.group().split("=")[-1]
            logger.debug(
                "No python version is specified under '%s'. Using the Python options specified under 'docker'.",
                environment_yml,
            )

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
        validator=attr.validators.optional(attr.validators.instance_of(ListStr)),
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
        validator=attr.validators.optional(attr.validators.instance_of(ListStr)),
    )
    find_links: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(ListStr)),
    )
    extra_index_url: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(ListStr)),
    )
    pip_args: t.Optional[str] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    wheels: t.Optional[t.List[str]] = attr.field(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(ListStr)),
    )

    def __attrs_post_init__(self):
        if self.requirements_txt and self.packages:
            logger.warning(
                "Build option python: 'requirements_txt={self.requirements_txt}' found, will ignore the option: 'packages=%s'.",
                self.requirements_txt,
                self.packages,
            )
        if self.no_index and (self.index_url or self.extra_index_url):
            logger.warning(
                "Build option python: 'no_index=%s' found, will ignore 'index_url' and 'extra_index_url' option when installing PyPI packages.",
                self.no_index,
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
        if self.wheels:
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
            args = ["--no-warn-script-location"]
            if pip_args:
                args.extend(pip_args)
            install_sh = (
                """\
#!/usr/bin/env bash
set -exuo pipefail

# Parent directory https://stackoverflow.com/a/246128/8643197
BASEDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"

PIP_ARGS=("""
                + " ".join(map(quote, args))
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
    pip3 install -r "$REQUIREMENTS_LOCK" "${PIP_ARGS[@]}"
else
    if [ -f "$REQUIREMENTS_TXT" ]; then
        echo "Installing pip packages from 'requirements.txt'.."
        pip3 install -r "$REQUIREMENTS_TXT" "${PIP_ARGS[@]}"
    fi
fi

# Install user-provided wheels
if [ -d "$WHEELS_DIR" ]; then
    echo "Installing wheels packaged in Bento.."
    pip3 install "$WHEELS_DIR"/*.whl "${PIP_ARGS[@]}"
fi

# Install the BentoML from PyPI if it's not already installed
if python3 -c "import bentoml" &> /dev/null; then
    existing_bentoml_version=$(python3 -c "import bentoml; print(bentoml.__version__)")
    if [ "$existing_bentoml_version" != "$BENTOML_VERSION" ]; then
        echo "WARNING: using BentoML version ${existing_bentoml_version}"
    fi
else
    pip3 install bentoml=="$BENTOML_VERSION"
fi
"""
            )
            f.write(install_sh)

        if self.requirements_txt is not None:
            from pip_requirements_parser import RequirementsFile

            requirements_txt = RequirementsFile.from_file(
                resolve_user_filepath(self.requirements_txt, build_ctx),
                include_nested=True,
            )
            # We need to make sure that we don't write any file references
            # back into the final `requirements.txt` file. We've already
            # resolved them and included their contents so we can discard
            # them.
            for option_line in requirements_txt.options:
                option_line.options.pop("constraints", None)
                option_line.options.pop("requirements", None)

            with bento_fs.open(
                fs.path.combine(py_folder, "requirements.txt"), "w"
            ) as f:
                f.write(requirements_txt.dumps(preserve_one_empty_line=True))
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
                    "--resolver=backtracking",
                ]
            )
            logger.info("Locking PyPI package versions.")
            cmd = [sys.executable, "-m", "piptools", "compile"]
            cmd.extend(pip_compile_args)
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                logger.error("Failed to lock PyPI packages: %s", e, exc_info=e)
                logger.error(
                    "Falling back to using the user-provided package requirement specifiers, which is equivalent to 'lock_packages=false'."
                )

    def with_defaults(self) -> PythonOptions:
        # Convert from user provided options to actual build options with default values
        defaults: dict[str, t.Any] = {}

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


if t.TYPE_CHECKING:
    OptionsCls = DockerOptions | CondaOptions | PythonOptions


def dict_options_converter(
    options_type: t.Type[OptionsCls],
) -> t.Callable[[OptionsCls | dict[str, t.Any]], t.Any]:
    def _converter(value: OptionsCls | dict[str, t.Any]) -> options_type:
        if value is None:
            return options_type()
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
    name: t.Optional[str] = None
    description: t.Optional[str] = None
    labels: t.Optional[t.Dict[str, t.Any]] = None
    include: t.Optional[t.List[str]] = None
    exclude: t.Optional[t.List[str]] = None
    docker: DockerOptions = attr.field(
        default=None,
        converter=dict_options_converter(DockerOptions),
    )
    python: PythonOptions = attr.field(
        default=None,
        converter=dict_options_converter(PythonOptions),
    )
    conda: CondaOptions = attr.field(
        default=None,
        converter=dict_options_converter(CondaOptions),
    )

    if t.TYPE_CHECKING:
        # NOTE: This is to ensure that BentoBuildConfig __init__
        # satisfies type checker. docker, python, and conda accepts
        # dict[str, t.Any] since our converter will handle the conversion.
        # There is no way to tell type checker signatures of the converter from attrs
        # if given attribute is alrady has a type annotation.
        def __init__(
            self,
            service: str,
            name: str | None = ...,
            description: str | None = ...,
            labels: dict[str, t.Any] | None = ...,
            include: list[str] | None = ...,
            exclude: list[str] | None = ...,
            docker: DockerOptions | dict[str, t.Any] | None = ...,
            python: PythonOptions | dict[str, t.Any] | None = ...,
            conda: CondaOptions | dict[str, t.Any] | None = ...,
        ) -> None:
            ...

    def __attrs_post_init__(self) -> None:
        use_conda = not self.conda.is_empty()
        use_cuda = self.docker.cuda_version is not None

        if use_cuda and use_conda:
            logger.warning(
                "BentoML does not support using both conda dependencies and setting a CUDA version for GPU. If you need both conda and CUDA, use a custom base image or create a dockerfile_template, see https://docs.bentoml.org/en/latest/concepts/bento.html#custom-base-image-advanced"
            )

        if self.docker.distro is not None:
            if use_conda and self.docker.distro not in get_supported_spec("miniconda"):
                raise BentoMLException(
                    f"{self.docker.distro} does not supports conda. BentoML will only support conda with the following distros: {get_supported_spec('miniconda')}."
                )

            spec = DistroSpec.from_options(self.docker, self.conda)
            if (
                self.docker.python_version is not None
                and self.docker.python_version not in spec.supported_python_versions
            ):
                raise BentoMLException(
                    f"{self.docker.python_version} is not supported for {self.docker.distro}. Supported python versions are: {', '.join(spec.supported_python_versions)}."
                )

            if self.docker.cuda_version is not None:
                if spec.supported_cuda_versions is None:
                    raise BentoMLException(
                        f"{self.docker.distro} does not support CUDA, while 'docker.cuda_version={self.docker.cuda_version}' is provided."
                    )
                if self.docker.cuda_version != "default" and (
                    self.docker.cuda_version not in spec.supported_cuda_versions
                ):
                    raise BentoMLException(
                        f"{self.docker.cuda_version} is not supported for {self.docker.distro}. Supported cuda versions are: {', '.join(spec.supported_cuda_versions)}."
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
            self.name,
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
        try:
            yaml.dump(bentoml_cattr.unstructure(self), stream)
        except yaml.YAMLError as e:
            logger.error("Error while deserializing BentoBuildConfig to yaml:")
            logger.error(e)
            raise


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
    name: t.Optional[str]
    description: t.Optional[str]
    labels: t.Dict[str, t.Any]
    include: t.List[str]
    exclude: t.List[str]
    docker: DockerOptions
    python: PythonOptions
    conda: CondaOptions
