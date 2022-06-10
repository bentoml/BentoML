import os
import re
import sys
import typing as t
import logging
import subprocess
from sys import version_info
from typing import TYPE_CHECKING

import fs
import attr
import yaml
import fs.copy
from dotenv import dotenv_values  # type: ignore

from .gen import generate_dockerfile
from ..utils import bentoml_cattr as cattr
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

if version_info >= (3, 8):
    from typing import Literal
else:  # pragma: no cover
    from typing_extensions import Literal

if TYPE_CHECKING:
    from attr import Attribute
    from fs.base import FS

if version_info >= (3, 10):
    from typing import TypeAlias
else:  # pragma: no cover
    from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)


# Docker defaults
DEFAULT_CUDA_VERSION = "11.6.2"
DEFAULT_DOCKER_DISTRO = "debian"


def _convert_python_version(py_version: t.Optional[str]) -> t.Optional[str]:
    if py_version is None:
        return None

    if not isinstance(py_version, str):
        py_version = str(py_version)

    match = re.match(r"^(\d+)\.(\d+)(?:|.(?:\d+|\w+))$", py_version)
    if match is None:
        raise InvalidArgument(
            f'Invalid build option: docker.python_version="{py_version}", python '
            f"version must follow standard python semver format, e.g. 3.7.10 ",
        )
    major, minor = match.groups()
    return f"{major}.{minor}"


def _convert_cuda_version(
    cuda_version: t.Optional[t.Union[str, int]]
) -> t.Optional[str]:
    if cuda_version is None or cuda_version == "":
        return None

    if not isinstance(cuda_version, str):
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
        res: t.Dict[str, t.Any] = {}
        for envar in envars:
            k, v = envar.split("=")
            res[k] = v
        return res
    else:
        return envars


def _envars_validator(
    _: t.Any,
    attribute: "Attribute[t.Union[str, t.List[str], t.Dict[str, t.Any]]]",
    value: t.Union[str, t.List[str], t.Dict[str, t.Any]],
) -> None:
    if isinstance(value, list):
        for envar in value:  # pragma: no cover
            #  we have tests for this cases, but pytest refuse to cover
            envars_rgx = re.compile(r"(^[A-Z_]+\w*)(\=)(.*)")
            if not envars_rgx.match(envar):
                raise InvalidArgument(
                    "All value in `env` list must follow format ENVAR=value"
                )
    elif isinstance(value, dict):
        uppercase_rgx = re.compile(r"(^[A-Z_]+\w*)")
        filter_non_envar: list[str] = []
        for k, v in value.items():
            if not uppercase_rgx.match(k):
                raise InvalidArgument(
                    "All keys in `env` dict must be UPPERCASE and start with a letter, e.g: `ENV_123: value`"
                )
            if not v:
                filter_non_envar.append(k)

        if len(filter_non_envar) > 0:
            logger.warning(
                f"`env` dict contains None value: {', '.join(filter_non_envar)}"
            )
    else:
        raise InvalidArgument(
            f"`env` must be either a list or a dict, got type {attribute.type} instead."
        )


@attr.frozen
class DockerOptions:
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

    def with_defaults(self) -> "DockerOptions":
        # Convert from user provided options to actual build options with default values
        defaults: t.Dict[str, t.Any] = {}

        if self.base_image is None:
            if self.distro is None:
                defaults["distro"] = DEFAULT_DOCKER_DISTRO
            if self.python_version is None:
                python_version = f"{version_info.major}.{version_info.minor}"
                defaults["python_version"] = python_version

        if self.system_packages is None:
            defaults["system_packages"] = None
        if self.env is None:
            defaults["env"] = {}
        if self.cuda_version is None:
            defaults["cuda_version"] = None
        if self.dockerfile_template is None:
            defaults["dockerfile_template"] = None

        return attr.evolve(self, **defaults)

    def write_to_bento(
        self, bento_fs: "FS", build_ctx: str, conda_cls: "CondaOptions"
    ) -> None:
        use_conda = False
        for attr_ in [a.name for a in attr.fields(conda_cls.__class__)]:
            if getattr(conda_cls, attr_) is not None:
                use_conda = True
                break

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


def _docker_options_structure_hook(d: t.Any, _: t.Type[DockerOptions]) -> DockerOptions:
    # Allow bentofile yaml to have either a str or list of str for these options
    if "env" in d and isinstance(d["env"], str):
        try:
            if not os.path.exists(d["env"]):
                raise FileNotFoundError
            d["env"] = dotenv_values(d["env"])
        except FileNotFoundError:
            raise BentoMLException(f"Invalid env file path: {d['env']}")

    return DockerOptions(**d)


cattr.register_structure_hook(DockerOptions, _docker_options_structure_hook)


if TYPE_CHECKING:
    CondaPipType = t.Dict[t.Literal["pip"], t.List[str]]
    DependencyType = t.List[t.Union[str, CondaPipType]]
else:
    DependencyType = list


def conda_dependencies_validator(
    _: t.Any, __: "Attribute[DependencyType]", value: DependencyType
) -> None:
    if not isinstance(value, list):
        raise InvalidArgument(
            f"Expected `conda.dependencies` to be type `list`, got type: `{type(value)}` instead"
        )
    else:
        conda_pip: "t.List[CondaPipType]" = [x for x in value if isinstance(x, dict)]
        if conda_pip:
            try:  # need to test this since conda didn't cover this :(
                if len(conda_pip) > 1:
                    raise InvalidArgument(
                        "Expected dictionary under `conda.dependencies` to ONLY have key `pip`"
                    )
                pip_list: t.List[str] = conda_pip[0]["pip"]
                if not all(isinstance(x, str) for x in pip_list):
                    raise InvalidArgument("Expected `pip` values to be type `str`")
            except KeyError:
                raise InvalidArgument(
                    "Conda dependencies can ONLY include `pip` value as a dictionary."
                )


@attr.frozen
class CondaOptions:

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

    def write_to_bento(self, bento_fs: "FS", build_ctx: str) -> None:
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
            deps_list.append({"pip": self.pip})

        if deps_list:
            yaml_content = {
                "dependencies": deps_list,
                "channels": ["defaults"] if self.channels is None else self.channels,
            }

            with bento_fs.open(fs.path.join(conda_folder, "environment.yml"), "w") as f:
                yaml.dump(yaml_content, f)

    def with_defaults(self) -> "CondaOptions":
        # Convert from user provided options to actual build options with default values
        defaults: t.Dict[str, t.Any] = {}

        if self.channels is not None:
            defaults["channels"] = self.channels + ["defaults"]

        return attr.evolve(self, **defaults)


def _conda_options_structure_hook(d: t.Any, _: t.Type[CondaOptions]) -> CondaOptions:
    pip_list: t.List[str]
    if "pip" in d and d["pip"] is not None:
        pip_list = d["pip"]
    else:
        pip_list = []
    if "dependencies" in d and d["dependencies"] is not None:
        deps: DependencyType = d["dependencies"]
        if not isinstance(deps, list):
            raise InvalidArgument(
                f"dependencies has to be type list, got type {type(d['dependencies'])}"
            )
        for dep in deps:
            if isinstance(dep, dict):
                assert "pip" in dep and len(dep.keys()) == 1
                pip_from_deps = dep.pop("pip")
                pip_list.extend(pip_from_deps)
        deps = list(filter(lambda x: not isinstance(x, dict), deps))
        d["dependencies"] = deps
    d["pip"] = None if not pip_list else pip_list

    return CondaOptions(**d)


cattr.register_structure_hook(CondaOptions, _conda_options_structure_hook)


@attr.frozen
class PythonOptions:
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
        validator=attr.validators.optional(
            lambda _, __, value: isinstance(value, str)
            and re.match(r"^(?:--)\w*", value)
        ),
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

    def write_to_bento(self, bento_fs: "FS", build_ctx: str) -> None:
        py_folder = fs.path.join("env", "python")
        wheels_folder = fs.path.join(py_folder, "wheels")
        bento_fs.makedirs(py_folder, recreate=True)

        # Save the python version of current build environment
        with bento_fs.open(fs.path.join(py_folder, "version.txt"), "w") as f:
            f.write(f"{version_info.major}.{version_info.minor}.{version_info.micro}")

        # Move over required wheel files
        # Note: although wheel files outside of build_ctx will also work, we should
        # discourage users from doing that
        if self.wheels is not None:
            bento_fs.makedirs(wheels_folder, recreate=True)
            for whl_file in self.wheels:  # pylint: disable=not-an-iterable
                whl_file = resolve_user_filepath(whl_file, build_ctx)
                copy_file_to_fs_folder(whl_file, bento_fs, wheels_folder)

        build_bentoml_editable_wheel(bento_fs.getsyspath(wheels_folder))

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
            with bento_fs.open(fs.path.combine(py_folder, "pip_args.txt"), "w") as f:
                f.write(" ".join(pip_args))

        if self.lock_packages:  # pragma: no cover
            # This section is not covered by tests and we relies
            # on jazzband/pip-tools test coverage

            # Note: "--allow-unsafe" is required for including setuptools in the
            # generated requirements.lock.txt file, and setuptool is required by
            # pyfilesystem2. Once pyfilesystem2 drop setuptools as dependency, we can
            # remove the "--allow-unsafe" flag here.

            # Note: "--generate-hashes" is purposefully not used here because it will
            # break if user includes PyPI package from version control system

            # changelog: move from fs.path.join to fs.path.combine
            # refers to https://docs.pyfilesystem.org/en/latest/reference/path.html#fs.path.combine
            pip_compile_in = bento_fs.getsyspath(
                fs.path.combine(py_folder, "requirements.txt")
            )
            pip_compile_out = bento_fs.getsyspath(
                fs.path.combine(py_folder, "requirements.lock.txt")
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
            cmd = [sys.executable, "-m", "piptools", "compile"]
            cmd.extend(pip_compile_args)
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed locking PyPI packages: {e}")
                logger.error(
                    "Falling back to using user-provided package requirement specifier, equivalent to `lock_packages=False`"
                )

    def with_defaults(self) -> "PythonOptions":
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


cattr.register_structure_hook(PythonOptions, _python_options_structure_hook)


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
    """This class is intended for modeling the bentofile.yaml file where user will
    provide all the options for building a Bento. All optional build options should be
    default to None so it knows which fields are NOT SET by the user provided config,
    which makes it possible to omitted unset fields when writing a BentoBuildOptions to
    a yaml file for future use. This also applies to nested options such as the
    DockerOptions class and the PythonOptions class.
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
        use_conda = False
        for attr_ in [a.name for a in attr.fields(self.conda.__class__)]:
            if getattr(self.conda, attr_) is not None:
                use_conda = True
                break
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
                # This is helpful when we support more CUDA version
                raise BentoMLException(  # pragma: no cover
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
                        raise BentoMLException(  # pragma: no cover
                            f"{self.docker.cuda_version} is not supported for {self.docker.distro}. Supported cuda versions are: {', '.join(_spec.supported_cuda_versions)}."
                        )

    def with_defaults(self) -> "FilledBentoBuildConfig":
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
    def from_yaml(cls, stream: t.TextIO) -> "BentoBuildConfig":
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise

        return cattr.structure(yaml_content, cls)

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
