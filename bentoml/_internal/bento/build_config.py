import os
import re
import typing as t
import logging
from sys import version_info as pyver

import fs
import attr
import yaml
import fs.copy
from fs.base import FS
from piptools.scripts.compile import cli as pip_compile_cli  # type: ignore

from ..tag import Tag
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from ..utils import copy_file_to_fs_folder
from .docker import ImageProvider
from ...exceptions import InvalidArgument
from .build_dev_bentoml_whl import build_bentoml_whl_to_target_if_in_editable_mode

logger = logging.getLogger(__name__)

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
PYTHON_MINOR_VERSION: str = f"{pyver.major}.{pyver.minor}"
PYTHON_SUPPORTED_VERSIONS: t.List[str] = ["3.7", "3.8", "3.9"]
DOCKER_SUPPORTED_DISTROS: t.List[str] = [
    "debian",
    "amazonlinux2",
    "alpine",
    "ubi8",
    "ubi7",
]
DOCKER_DEFAULT_DISTRO = "debian"

if PYTHON_MINOR_VERSION not in PYTHON_SUPPORTED_VERSIONS:
    logger.warning(
        "BentoML may not work well with current python version %s, supported python versions are: %s",
        PYTHON_MINOR_VERSION,
        ",".join(PYTHON_SUPPORTED_VERSIONS),
    )


def _convert_python_version(py_version: t.Optional[str]) -> t.Optional[str]:
    if py_version is None:
        return None

    match = re.match(r"^(\d+).(\d+)", py_version)
    if match is None:
        raise InvalidArgument(
            f'Invalid build option: docker.python_version="{py_version}", python '
            f"version must follow standard python semver format, e.g. 3.7.10 ",
        )
    major, minor = match.groups()
    return f"{major}.{minor}"


@attr.frozen
class DockerOptions:
    # Options for choosing a BentoML built-in docker images
    distro: str = attr.field(
        validator=attr.validators.optional(
            attr.validators.in_(DOCKER_SUPPORTED_DISTROS)
        ),
        default=None,
    )
    python_version: t.Optional[str] = attr.field(
        converter=_convert_python_version,
        default=None,
        validator=attr.validators.optional(
            attr.validators.in_(PYTHON_SUPPORTED_VERSIONS)
        ),
    )
    gpu: t.Optional[bool] = None
    devel: t.Optional[bool] = None

    # A python or shell script that executes during docker build time
    setup_script: t.Optional[str] = None

    # A user-provided custom docker image
    base_image: t.Optional[str] = None

    def __attrs_post_init__(self):
        if self.base_image is not None:
            if self.distro is not None:
                logger.warning(
                    "docker base_image %s is used, 'distro=%s' option is ignored",
                    self.base_image,
                    self.distro,
                )
            if self.python_version is not None:
                logger.warning(
                    "docker base_image %s is used, 'python=%s' option is ignored",
                    self.base_image,
                    self.python_version,
                )
            if self.gpu is not None:
                logger.warning(
                    "docker base_image %s is used, 'gpu=%s' option is ignored",
                    self.base_image,
                    self.gpu,
                )

    def with_defaults(self) -> "DockerOptions":
        # Convert from user provided options to actual build options with default values
        update_defaults = {}

        if self.base_image is None:
            if self.distro is None:
                update_defaults["distro"] = DOCKER_DEFAULT_DISTRO
            if self.python_version is None:
                update_defaults["python_version"] = PYTHON_MINOR_VERSION
            if self.gpu is None:
                update_defaults["gpu"] = False

        return attr.evolve(self, **update_defaults)

    def get_base_image_tag(self):
        if self.base_image is None:
            if self.distro is None:
                raise KeyError("distro not set, can't get base image tag")
            base_image = repr(
                ImageProvider(self.distro, self.python_version, self.gpu, self.devel)
            )
            return base_image
        else:
            return self.base_image

    def write_to_bento(self, bento_fs: FS, build_ctx: str):
        docker_folder = fs.path.join("env", "docker")
        bento_fs.makedirs(docker_folder, recreate=True)
        dockerfile = fs.path.join(docker_folder, "Dockerfile")
        template_file = os.path.join(
            os.path.dirname(__file__), "docker", "Dockerfile.template"
        )
        with open(template_file, "r", encoding="utf-8") as f:
            dockerfile_template = f.read()

        with bento_fs.open(dockerfile, "w") as dockerfile:
            dockerfile.write(
                dockerfile_template.format(base_image=self.get_base_image_tag())
            )

        for filename in ["init.sh", "entrypoint.sh"]:
            copy_file_to_fs_folder(
                os.path.join(os.path.dirname(__file__), "docker", filename),
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


@attr.frozen
class CondaOptions:
    environment_yml: t.Optional[str] = None
    channels: t.Optional[t.List[str]] = None
    dependencies: t.Optional[t.List[str]] = None
    pip: t.Optional[t.List[str]] = None  # list of pip packages to install via conda

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

    def write_to_bento(self, bento_fs: FS, build_ctx: str):
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
                dst_filename="environment_yml",
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
        # Convert from user provided options to actual build options with default values
        update_defaults = {}

        if self.channels is None:
            update_defaults["channels"] = ["defaults"]

        return attr.evolve(self, **update_defaults)


@attr.frozen
class PythonOptions:
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
                f'Build option python: requirements_txt="{self.requirements_txt}" found,'
                f' this will ignore the option: packages="{self.packages}"'
            )
        if self.no_index and (self.index_url or self.extra_index_url):
            logger.warning(
                "Build option python.no_index=True found, this will ignore index_url"
                " and extra_index_url option when installing PyPI packages"
            )

    def write_to_bento(self, bento_fs: FS, build_ctx: str):
        py_folder = fs.path.join("env", "python")
        wheels_folder = fs.path.join(py_folder, "wheels")
        bento_fs.makedirs(py_folder, recreate=True)

        # Save the python version of current build environment
        with bento_fs.open(fs.path.join(py_folder, "version.txt"), "w") as f:
            f.write(PYTHON_VERSION)

        # Move over required wheel files
        # Note: although wheel files outside of build_ctx will also work, we should
        # discourage users from doing that
        if self.wheels is not None:
            for whl_file in self.wheels:  # pylint: disable=not-an-iterable
                whl_file = resolve_user_filepath(whl_file, build_ctx)
                copy_file_to_fs_folder(whl_file, bento_fs, wheels_folder)

        # If BentoML is installed in editable mode, build bentoml whl and save to Bento
        build_bentoml_whl_to_target_if_in_editable_mode(
            bento_fs.getsyspath(wheels_folder)
        )

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
        # Convert from user provided options to actual build options with default values
        update_defaults = {}

        if self.requirements_txt is None:
            if self.lock_packages is None:
                update_defaults["lock_packages"] = True

        return attr.evolve(self, **update_defaults)


def _python_options_structure_hook(d: t.Any, _: t.Type[PythonOptions]):
    # Allow bentofile yaml to have either a str or list of str for these options
    for field in ["trusted_host", "find_links", "extra_index_url"]:
        if field in d and isinstance(d[field], str):
            d[field] = [d[field]]

    return PythonOptions(**d)


bentoml_cattr.register_structure_hook(PythonOptions, _python_options_structure_hook)


def _dict_arg_converter(  # type: ignore
    options_type: t.Type[t.Union[DockerOptions, CondaOptions, PythonOptions]]
):
    def _converter(
        value: t.Optional[
            t.Union[DockerOptions, CondaOptions, PythonOptions, t.Dict[str, t.Any]]
        ]
    ) -> t.Optional[options_type]:
        if isinstance(value, dict):
            return options_type(**value)
        return value

    return _converter  # type: ignore


def _additional_models_converter(
    tags: t.Optional[t.List[t.Union[str, Tag]]]
) -> t.Optional[t.List[Tag]]:
    if tags is None:
        return tags

    return list(map(Tag.from_taglike, tags))


@attr.define(frozen=True)
class BentoBuildConfig:
    """This class is intended for modeling the bentofile.yaml file where user will
    provide all the options for building a Bento. All optional build options should be
    default to None so it knows which fields are NOT SET by the user provided config,
    which makes it possible to omitted unset fields when writing a BentoBuildOptions to
    a yaml file for future use. This also applies to nested options such as the
    DockerOptions class and the PythonOptions class.
    """

    service: str  # Import Str of target service to build
    description: t.Optional[str] = None
    labels: t.Optional[t.Dict[str, t.Any]] = None
    include: t.Optional[t.List[str]] = None
    exclude: t.Optional[t.List[str]] = None
    additional_models: t.Optional[t.List[Tag]] = attr.field(
        converter=_additional_models_converter,
        default=None,
    )
    docker: t.Optional[DockerOptions] = attr.field(
        default=None,
        converter=_dict_arg_converter(DockerOptions),
    )
    python: t.Optional[PythonOptions] = attr.field(
        default=None,
        converter=_dict_arg_converter(PythonOptions),
    )
    conda: t.Optional[CondaOptions] = attr.field(
        default=None,
        converter=_dict_arg_converter(CondaOptions),
    )

    def with_defaults(self) -> "FilledBentoBuildConfig":
        """Convert from user provided options to actual build options will defaults
        values filled in

        Returns:
            BentoBuildConfig: a new copy of self, with default values filled
        """

        return FilledBentoBuildConfig(
            self.service,
            self.description,
            {} if self.labels is None else self.labels,
            ["*"] if self.include is None else self.include,
            [] if self.exclude is None else self.exclude,
            [] if self.additional_models is None else self.additional_models,
            (DockerOptions() if self.docker is None else self.docker).with_defaults(),
            (PythonOptions() if self.python is None else self.python).with_defaults(),
            (CondaOptions() if self.conda is None else self.conda).with_defaults(),
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

    def to_yaml(self, stream: t.TextIO):
        # TODO: Save BentoBuildOptions to a yaml file
        # This is reserved for building interactive build file creation CLI
        raise NotImplementedError


class FilledBentoBuildConfig(BentoBuildConfig):
    service: str
    description: t.Optional[str]
    labels: t.Dict[str, t.Any]
    include: t.List[str]
    exclude: t.List[str]
    additional_models: t.List[Tag]
    docker: DockerOptions
    python: PythonOptions
    conda: CondaOptions
