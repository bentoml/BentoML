#!/usr/bin/env python3
import errno
import inspect
import json
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from typing import Dict, Optional
from dotenv import load_dotenv

import absl.logging
import docker
from absl import flags, app
from cerberus import Validator
from glom import glom, Path, PathAccessError, Assign, PathAssignError
from jinja2 import Environment
from ruamel import yaml

load_dotenv()


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31"
    reset = "\x1b[0m"
    _format = "[%(levelname)s::L%(lineno)d] %(funcName)s :: %(message)s"

    FORMATS = {
        logging.DEBUG: yellow + _format + reset,
        logging.INFO: magenta + _format + reset,
        logging.WARNING: blue + _format + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: bold_red + _format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# quick hack to overwrite absl.logging format
absl.logging.get_absl_handler().setFormatter(ColoredFormatter())
logger = absl.logging.get_absl_logger()

DOCKERFILE_PREFIX = ".Dockerfile.j2"
REPO_PREFIX = ".repo.j2"
DOCKERFILE_ALLOWED_NAMES = ["base", "cudnn", "devel", "runtime"]

SUPPORTED_PYTHON_VERSION = ["3.6", "3.7", "3.8"]
SUPPORTED_OS = ["ubuntu", "debian", "centos", "alpine", "amazonlinux"]

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"
)

# defined global vars.
FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "push_to_hub",
    False,
    "Whether to upload images to given registries.",
    short_name="pth",
)
flags.DEFINE_integer("timeout", 3600, "Timeout for docker build", short_name="t")

# CLI-related
flags.DEFINE_boolean(
    "dump_metadata",
    False,
    "Whether to dump all tags metadata to file.",
    short_name="dm",
)
flags.DEFINE_boolean(
    "dry_run",
    False,
    "Whether to dry run. This won't create Dockerfile.",
    short_name="dr",
)
flags.DEFINE_boolean(
    "stop_at_generate", False, "Whether to just generate dockerfile.", short_name="sag"
)
flags.DEFINE_boolean("build_images", False, "Whether to build images.", short_name="bi")
flags.DEFINE_boolean(
    "stop_on_failure",
    False,
    "Stop processing tags if any one build fails. If False or not specified, failures are reported but do not affect "
    "the other images.",
    short_name="sof",
)
# directory and files
flags.DEFINE_string(
    "dockerfile_dir",
    "./generated",
    "path to generated Dockerfile. Existing files will be deleted with new Dockerfiles",
    short_name="dd",
)
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file", short_name="mf")

flags.DEFINE_string(
    "bentoml_version", None, "BentoML release version", required=True, short_name="bv"
)

flags.DEFINE_string("cuda_version", "11.3.1", "Define CUDA version", short_name="cv")

flags.DEFINE_string(
    "python_version",
    None,
    "OPTIONAL: Python version to build Docker images (useful when developing).",
    short_name="pv",
)

SPEC_SCHEMA = """
--- 
repository:
  type: dict
  keysrules:
    type: string
  valuesrules:
    type: dict
    schema:
      only_if:
        required: true
        env_vars: true
        type: string
      pwd:
        dependencies: only_if
        env_vars: true
        type: string
      user:
        dependencies: only_if
        env_vars: true
        type: string
      registry:
        keysrules:
          check_with: packages
          type: string
        type: dict
        valuesrules:
          type: string

packages:
  type: dict
  keysrules:
    type: string
  valuesrules:
    type: dict
    allowed: [devel, cudnn, runtime]
    schema:
      runtime:
        required: true
        type: [string, list]
        check_with: dists_defined
      devel:
        type: list
        dependencies: runtime
        check_with: dists_defined
        schema:
          type: string
      cudnn:
        type: [string, list]
        check_with: dists_defined
        dependencies: runtime

cuda_dependencies:
  type: dict
  nullable: false
  keysrules:
    type: string
    regex: '(\d{1,2}\.\d{1}\.\d)?$'
  valuesrules:
    type: dict
    keysrules:
      type: string
    valuesrules:
      type: string

release_spec:
  required: true
  type: dict
  schema:
    add_to_tags:
      type: string
      required: true
    templates_dir:
      type: string
    base_image:
      type: string
    header:
      type: string
    cuda_prefix_url:
      type: string
      dependencies: cuda
    cuda_requires:
      type: string
      dependencies: cuda
    needs_deps_image:
      type: boolean
    envars:
      type: list
      default: []
      schema:
        check_with: args_format
        type: string
    cuda:
      type: dict

releases:
  type: dict
  keysrules:
    supported_dists: true
    type: string
  valuesrules:
    type: dict
    keysrules:
      regex: '([\d\.]+)'
    valuesrules:
      type: dict
    
"""


class MetadataSpecValidator(Validator):
    """
    Custom cerberus validator for BentoML metadata spec.

    Refers to https://docs.python-cerberus.org/en/stable/customize.html#custom-rules.
    This will add two rules:
    - args should have the correct format: ARG=foobar as well as correct ENV format.
    - releases should be defined under SUPPORTED_OS

    Args:
        packages: bentoml release packages, model-server, yatai-service, etc
    """

    # TODO: custom check for releases.valuesrules

    def __init__(self, *args, **kwargs):
        if "packages" in kwargs:
            self.packages = kwargs["packages"]
        if "releases" in kwargs:
            self.releases = kwargs["releases"]
        super(Validator, self).__init__(*args, **kwargs)

    def _check_with_packages(self, field, value):
        """
        Check if registry is defined in packages.
        """
        if value not in self.packages:
            self._error(field, f"{field} is not defined under packages")

    def _check_with_args_format(self, field, value):
        """
        Check if value has correct envars format: ARG=foobar
        This will be parsed at runtime when building docker images.
        """
        if isinstance(value, str):
            if "=" not in value:
                self._error(
                    field, f"{value} should have format ARG=foobar",
                )
            else:
                envars, _, _ = value.partition("=")
                self._validate_env_vars(True, field, envars)
        if isinstance(value, list):
            for v in value:
                self._check_with_args_format(field, v)

    def _check_with_dists_defined(self, field, value):
        if isinstance(value, list):
            for v in value:
                self._check_with_dists_defined(field, v)
        else:
            if value not in self.releases:
                self._error(field, f"{field} is not defined under releases")

    def _validate_env_vars(self, env_vars, field, value):
        """
        Validate if given is a environment variable.

        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if isinstance(value, str):
            if env_vars and not value.isupper():
                self._error(field, f"{value} cannot be parsed to envars.")
        else:
            self._error(field, f"{value} cannot be parsed as string.")

    def _validate_supported_dists(self, supported_dists, field, value):
        """
        Validate if given is a supported OS.

        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if isinstance(value, str):
            if supported_dists and value not in SUPPORTED_OS:
                self._error(
                    field,
                    f"{value} is not defined in SUPPORTED_OS. If you are adding a new distros make "
                    f"sure to add it to SUPPORTED_OS",
                )
        if isinstance(value, list):
            for v in value:
                self._validate_supported_dists(supported_dists, field, v)


def auth_registries(repository, docker_client):
    repos = {}
    for registry, metadata in repository.items():
        if not os.getenv(metadata['only_if']) or not os.getenv(metadata['user']):
            logger.critical(
                f"{registry}: {metadata['only_if']} or {metadata['user']} is not set or not found."
            )
        repos[registry] = {
            'user': os.getenv(metadata['user']),
            'pwd': os.getenv(metadata['pwd']),
            'package_registry': list(metadata['registry'].values()),
        }

    for repo, data in repos.items():
        logger.info(f"> Logging into Docker registry {repo} with credentials...")
        docker_client.login(username=data['user'], password=data['pwd'], registry=repo)
        repos[repo] = {'package_registry': data['package_registry']}
    return repos


def upload_in_background(docker_client, image, registry, tag):
    """Upload a docker image (to be used by multiprocessing)."""
    image.tag(registry, tag=tag)
    logger.debug(docker_client.images.push(registry, tag=tag))


def flatten(arr):
    for it in arr:
        if isinstance(it, Iterable) and not isinstance(it, str):
            for x in flatten(it):
                yield x
        else:
            yield it


def load_manifest_yaml(file):
    with open(file, "r") as input_file:
        manifest = yaml.safe_load(input_file)

    v_schema = yaml.safe_load(SPEC_SCHEMA)
    v = MetadataSpecValidator(
        v_schema, packages=manifest["packages"], releases=manifest["releases"]
    )

    if not v.validate(manifest):
        logger.error(f"{file} is invalid. Errors as follow:\n")
        logger.error(yaml.dump(v.errors, indent=2))
        exit(1)

    release_spec = v.normalized(manifest)
    return release_spec


def validate_template_name(names):
    """Validate our input file name for Dockerfile templates."""
    if isinstance(names, str):
        if REPO_PREFIX in names:
            # ignore .repo.j2 for centos
            pass
        elif DOCKERFILE_PREFIX in names:
            names, _, _ = names.partition(DOCKERFILE_PREFIX)
            if names not in DOCKERFILE_ALLOWED_NAMES:
                logger.critical(
                    f"Unable to parse {names}. Dockerfile allowed name: {', '.join(DOCKERFILE_ALLOWED_NAMES)} "
                )
        else:
            logger.warning(
                f"{inspect.currentframe().f_code.co_name} is being used to validate {names}, which has unknown "
                f"extensions, thus will have no effects. "
            )
            pass
    elif isinstance(names, list):
        for _name in names:
            validate_template_name(_name)
    else:
        raise TypeError(f"{names} has unknown type {type(names)}")


def get_data(obj, *path, skip=False):
    """
    Get data from the object by dotted path.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        data = glom(obj, Path(*path))
    except PathAccessError:
        if skip:
            return
        logger.exception(
            f"Exception occurred in get_data, unable to retrieve from {' '.join(*path)}"
        )
    else:
        return data


def update_data(obj, value, *path):
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(Path(*path), value))
    except PathAssignError:
        logger.exception(
            f"Exception occurred in update_data, unable to update {Path(*path)} with {value}"
        )


def _build_release_args(bentoml_version):
    """
    Create a dictionary of args that will be parsed during runtime

    Args:
        bentoml_version: version of BentoML releases

    Returns:
        Dict of args
    """
    # we will update PYTHON_VERSION when build process is invoked
    args = {"PYTHON_VERSION": "", "BENTOML_VERSION": bentoml_version}

    def update_args_dict(updater):
        # updater will be distro_release_spec['envars'] (e.g)
        # Args will have format ARG=foobar
        if isinstance(updater, str):
            _args, _, _value = updater.partition("=")
            args[_args] = _value
            return
        elif isinstance(updater, list):
            for v in updater:
                update_args_dict(v)
        elif isinstance(updater, dict):
            for _args, _value in updater.items():
                args[_args] = _value
        else:
            logger.error(f"cannot add to args dict with unknown type {type(updater)}")
        return args

    return update_args_dict


def _build_release_tags(build_ctx, release_type, base_tags):
    """
    Our tags will have format: {bentoml_release_version}-{python_version}-{distros}-{image_type}
    eg:
    - python3.8-ubuntu20.04-base -> contains conda and python preinstalled
    - devel-python3.8-ubuntu20.04
    - 0.13.0-python3.7-centos8-{runtime,cudnn}
    """

    _python_version = get_data(build_ctx, "envars", "PYTHON_VERSION")

    # check if PYTHON_VERSION is set.
    if not _python_version:
        logger.critical("PYTHON_VERSION should be set before generating Dockerfile.")
    if _python_version not in SUPPORTED_PYTHON_VERSION:
        logger.critical(
            f"{_python_version} is not supported. Supported versions: {SUPPORTED_PYTHON_VERSION}"
        )

    _tag = ""
    core_string = "-".join([f"python{_python_version}", build_ctx["add_to_tags"]])

    try:
        base_tag = "-".join([core_string, base_tags])
    except KeyError:
        base_tag = core_string

    if release_type == "devel":
        _tag = "-".join([release_type, core_string])
    elif release_type == "base":
        pass
    else:
        _tag = "-".join(
            [
                get_data(build_ctx, "envars", "BENTOML_VERSION"),
                core_string,
                release_type,
            ]
        )
    return _tag, base_tag


def _update_build_ctx(ctx_dict, spec_dict: Dict, *ignore_keys):
    # flatten release_spec to template context.
    for k, v in spec_dict.items():
        if k in list(flatten([*ignore_keys])):
            continue
        ctx_dict[k] = v


def _add_base_release_type(build_ctx, release_type: list):
    try:
        return release_type + [build_ctx["base_tags"]]
    except KeyError:
        # when base_tags is either not set or available
        return release_type


# noinspection PyPep8Naming
class cached_property(property):
    """Direct ports from bentoml/utils/__init__.py"""

    class _Missing(object):
        def __repr__(self):
            return "no value"

        def __reduce__(self):
            return "_missing"

    _missing = _Missing()

    def __init__(self, func, name=None, doc=None):
        super().__init__(doc)
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, types=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class TemplatesMixin(object):
    """Mixin for generating Dockerfile from templates."""

    _build_ctx = {}
    # _build_path will consist of
    #   key: package_tag
    #   value: path to dockerfile
    _build_path = {}

    _template_env = Environment(
        extensions=["jinja2.ext.do"], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec, cuda_version):
        # set private class attributes to be manifest.yml keys
        for keys, values in release_spec.items():
            super().__setattr__(f"_{keys}", values)

        # setup default keys for build context
        _update_build_ctx(self._build_ctx, self._release_spec, "cuda")

        # setup cuda_version and cuda_components
        if self._check_cuda_version(cuda_version):
            self._cuda_version = cuda_version
        else:
            raise RuntimeError(
                f"CUDA {cuda_version} is either not supported or not defined in {self._cuda_dependencies.keys()}"
            )
        self._cuda_components = get_data(self._cuda_dependencies, self._cuda_version)

    def __call__(self, dry_run=False, *args, **kwargs):
        if FLAGS.python_version is not None:
            logger.debug("--python_version defined, ignoring SUPPORTED_PYTHON_VERSION")
            supported_python = [FLAGS.python_version]
        else:
            supported_python = SUPPORTED_PYTHON_VERSION

        if not dry_run:
            # Considered generated directory is ephemeral
            logger.info(f"Removing dockerfile dir: {FLAGS.dockerfile_dir}\n")
            shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
            try:
                os.makedirs(FLAGS.dockerfile_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            self.generate_dockerfiles(
                target_dir=FLAGS.dockerfile_dir, python_version=supported_python
            )

    @cached_property
    def build_path(self):
        return self._build_path

    def _cudnn_versions(self):
        for k, v in self._cuda_components.items():
            if k.startswith("cudnn") and v:
                return k, v

    def _check_cuda_version(self, cuda_version):
        # get defined cuda version for cuda
        return True if cuda_version in self._cuda_dependencies.keys() else False

    def render_dockerfile_from_templates(
        self,
        input_tmpl_path: pathlib.Path,
        output_path: pathlib.Path,
        ctx: Dict,
        tags: Optional[Dict[str, str]] = None,
    ):
        with open(input_tmpl_path, "r") as inf:
            logger.debug(f">>>> Processing template: {input_tmpl_path}\n")
            name = input_tmpl_path.name
            if DOCKERFILE_PREFIX not in name:
                output_name = name[: -len(".j2")]
            else:
                output_name = DOCKERFILE_PREFIX.split(".")[1]

            template = self._template_env.from_string(inf.read())
            if not output_path.exists():
                logger.debug(f">>>> Rendering to {output_path}/{output_name}")
                output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_path}/{output_name}", "w") as ouf:
                ouf.write(template.render(metadata=ctx, tags=tags))
            ouf.close()
        inf.close()

    def generate_distro_release_mapping(self, package):
        """Return a dict containing keys as distro and values as release type (devel, cudnn, runtime)"""
        releases = defaultdict(list)
        for k, v in self._packages[package].items():
            _v = list(flatten(v))
            for distro in _v:
                releases[distro].append(k)
        return releases

    def generate_template_context(
        self, distro, distro_version, distro_release_spec, python_version
    ):
        """
        Generate template context for each distro releases.

        build context structure:
            {
              "templates_dir": "",
              "base_image": "",
              "add_to_tags": "",
              "envars": {},
              "cuda_prefix_url": "",
              "cuda": {
                "ml_repo": "",
                "base_repo": "",
                "version": {},
                "components": {}
              }
            }
        """
        _build_ctx = deepcopy(self._build_ctx)

        cuda_regex = re.compile(r"cuda*")
        cudnn_regex = re.compile(r"(cudnn)(\w+)")

        _update_build_ctx(
            _build_ctx, get_data(self._releases, distro, distro_version), "cuda"
        )

        _build_ctx["envars"] = _build_release_args(FLAGS.bentoml_version)(
            distro_release_spec["envars"]
        )

        # setup base_tags if needed
        if _build_ctx["needs_deps_image"] is True:
            _build_ctx["base_tags"] = "base"
        _build_ctx.pop("needs_deps_image")

        # setup python version
        update_data(
            _build_ctx, python_version, "envars", "PYTHON_VERSION",
        )

        # setup cuda deps
        if _build_ctx["cuda_prefix_url"]:
            major, minor, _ = self._cuda_version.rsplit(".")
            cudnn_version_name, cudnn_release_version = self._cudnn_versions()
            cudnn_groups = cudnn_regex.match(cudnn_version_name).groups()
            cudnn_namespace, cudnn_major_version = [
                cudnn_groups[i] for i in range(len(cudnn_groups))
            ]
            _build_ctx["cuda"] = {
                "ml_repo": NVIDIA_ML_REPO_URL.format(_build_ctx["cuda_prefix_url"]),
                "base_repo": NVIDIA_REPO_URL.format(_build_ctx["cuda_prefix_url"]),
                "version": {
                    "release": self._cuda_version,
                    "major": major,
                    "minor": minor,
                    "shortened": ".".join([major, minor]),
                },
                cudnn_namespace: {
                    "version": cudnn_release_version,
                    "major_version": cudnn_major_version,
                },
                "components": self._cuda_components,
            }
            _build_ctx.pop("cuda_prefix_url")
        else:
            logger.warning(f">>>> CUDA is disabled for {distro}. Skipping...")
            for key in list(_build_ctx.keys()):
                if cuda_regex.match(key):
                    _build_ctx.pop(key)

        return _build_ctx

    def generate_dockerfiles(self, target_dir, python_version):
        """
        Workflow:
        walk all templates dir from given context

        our targeted generate directory should look like:
        generated
            -- packages (model-server, yatai-service)
                -- python_version (3.8,3.7,etc)
                    -- {distro}{distro_version} (ubuntu20.04, ubuntu18.04, centos8, centos7, amazonlinux2, etc)
                        -- runtime
                            -- Dockerfile
                        -- devel
                            -- Dockerfile
                        -- cudnn
                            -- Dockerfile
                        -- Dockerfile (for base deps images)

        Args:
            target_dir: parent directory of generated Dockerfile. Default: ./generated.
            python_version: target list of supported python version. Overwrite default supports list if
                            FLAGS.python_version is given.

        """
        _tag_metadata = defaultdict(list)

        logger.info(f"Generating dockerfile to {target_dir}...\n")
        for package in self._packages.keys():
            # update our packages ctx.
            logger.info(f"> Working on {package}")
            update_data(self._build_ctx, package, "package")
            for _py_ver in python_version:
                logger.info(f">> For python{_py_ver}")
                _distro_release_mapping = self.generate_distro_release_mapping(package)
                for distro, _release_type in _distro_release_mapping.items():
                    for distro_version, distro_spec in self._releases[distro].items():
                        # setup our build context
                        _build_ctx = self.generate_template_context(
                            distro, distro_version, distro_spec, _py_ver
                        )

                        logger.debug(f">>> Generating: {distro}{distro_version}")

                        # validate our template directory.
                        for _, _, files in os.walk(_build_ctx["templates_dir"]):
                            validate_template_name(files)

                        for _re_type in _add_base_release_type(
                            _build_ctx, _release_type
                        ):
                            # setup filepath
                            input_tmpl_path = pathlib.Path(
                                _build_ctx["templates_dir"],
                                f"{_re_type}{DOCKERFILE_PREFIX}",
                            )

                            base_output_path = pathlib.Path(
                                target_dir,
                                package,
                                _py_ver,
                                f"{distro}{distro_version}",
                            )

                            # handle cuda.repo and nvidia-ml.repo
                            if (
                                "rhel" in _build_ctx["templates_dir"]
                                and _re_type == "cudnn"
                            ):
                                for _n in ["nvidia-ml", "cuda"]:
                                    repo_tmpl_path = pathlib.Path(
                                        _build_ctx["templates_dir"],
                                        f"{_n}{REPO_PREFIX}",
                                    )
                                    repo_output_path = pathlib.Path(
                                        base_output_path, _re_type
                                    )
                                    self.render_dockerfile_from_templates(
                                        repo_tmpl_path, repo_output_path, _build_ctx
                                    )

                            # generate our image tag.
                            tag, basetag = _build_release_tags(
                                _build_ctx, _re_type, _build_ctx['base_tags']
                            )
                            tag_ctx = {"basename": basetag}

                            if _re_type != "base":
                                output_path = pathlib.Path(base_output_path, _re_type)
                            else:
                                output_path = base_output_path

                            # setup directory correspondingly, and add these paths
                            # with correct tags name under self._build_path
                            if tag:
                                tag_keys = f"{_build_ctx['package']}:{tag}"
                            else:
                                tag_keys = f"{_build_ctx['package']}:{basetag}"

                            _tag_metadata[tag_keys].append(_build_ctx)
                            self._build_path[tag_keys] = str(
                                pathlib.Path(output_path, "Dockerfile")
                            )
                            # generate our Dockerfile from templates.
                            self.render_dockerfile_from_templates(
                                input_tmpl_path=input_tmpl_path,
                                output_path=output_path,
                                ctx=_build_ctx,
                                tags=tag_ctx,
                            )

        if FLAGS.dump_metadata:
            file = "./metadata.json"
            logger.info(
                f"> --dump_metadata is specified. Dumping metadata to {file}..."
            )
            with open(file, "w") as ouf:
                ouf.write(json.dumps(_tag_metadata, indent=2))
            ouf.close()
            del _tag_metadata


def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    # Parse specs from manifest.yml and validate it.
    release_spec = load_manifest_yaml(FLAGS.manifest_file)
    docker_client = docker.from_env()

    # Login Docker credentials
    repos = auth_registries(release_spec["repository"], docker_client)

    # generate templates to correct directory.
    tmpl_mixin = TemplatesMixin(release_spec, cuda_version=FLAGS.cuda_version)
    tmpl_mixin(dry_run=FLAGS.dry_run)

    if FLAGS.stop_at_generate:
        logger.debug("--stop_at_generate is specified. Stopping now...")
        return

    # We will build and push if args is parsed. Establish some variables.
    push_tags, tag_failed = {}, False
    logs, failed_tags, succeeded_tags = [], [], []

    # need --build_images to run this segment.
    logger.info('-'*100+"\n")
    if FLAGS.build_images:
        for image_tag, dockerfile_path in tmpl_mixin.build_path.items():
            try:
                logger.info(f"> building {image_tag} from {dockerfile_path}")
                resp = docker_client.api.build(
                    timeout=FLAGS.timeout,
                    path=".",
                    nocache=False,
                    dockerfile=dockerfile_path,
                    tag=image_tag,
                )
                last_event, image_id = None, None
                while True:
                    try:
                        output = next(resp).decode('utf-8')
                        json_output = json.loads(output.strip('\r\n'))
                        if 'stream' in json_output:
                            logger.info(json_output['stream'])
                            match = re.search(
                                r'(^Successfully built |sha256:)([0-9a-f]+)$',
                                json_output['stream'],
                            )
                            if match:
                                image_id = match.group(2)
                            last_event = json_output['stream']
                            logs.append(json_output)
                    except StopIteration:
                        logger.debug(f">>> Successfully built {image_tag}.")
                        break
                    except ValueError:
                        logger.error(f">>> Errors while building image:\n{output}")
                if image_id:
                    push_tags[image_tag] = docker_client.images.get(image_id)
                else:
                    raise docker.errors.BuildError(last_event or 'Unknown', logs)
            except docker.errors.BuildError as e:
                logger.error(f">> Failed to build {image_tag} :\n{e.msg}")
                for line in e.build_log:
                    if 'stream' in line:
                        logger.error(line['stream'].strip())
                tag_failed = True
                failed_tags.append(image_tag)
                if FLAGS.stop_on_failure:
                    logger.fatal('>> ABORTING due to --stop_on_failure!')
            if not tag_failed:
                succeeded_tags.append(image_tag)
    else:
        logger.info(">>> --build_images is not specified. Skip building images...")

    logger.info(push_tags)
    # Push process
    if FLAGS.push_to_hub:
        for repo, registry in repos.items():
            for image_tag, image in push_tags.items():
                # push newly created image to registry.
                if tag_failed:
                    continue

                logger.info(f"> Uploading {image} to {repo}")
                if not FLAGS.dry_run:
                    p = multiprocessing.Process(
                        target=upload_in_background,
                        args=(docker_client, image, registry, image_tag),
                    )
                    p.start()
    else:
        logger.info(">>> --push_to_hub is not specified. Skip pushing images...")

    if failed_tags:
        logger.fatal(
            f"> Some tags failed to build or failed testing, check scroll back for errors: {','.join(failed_tags)}"
        )
    if succeeded_tags:
        logger.info("Success tags:\n")
        for tag in succeeded_tags:
            print(f"\n{tag}")


if __name__ == "__main__":
    app.run(main)