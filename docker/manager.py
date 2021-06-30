#!/usr/bin/env python3
import itertools
import json
import multiprocessing
import os
import pathlib
import re
import shutil
import time
from collections import defaultdict
from typing import Optional, Union, Dict, MutableMapping, List

import absl.logging as logging
import docker
from absl import app
from cerberus import Validator
from dotenv import load_dotenv
from glom import glom, Path, PathAccessError, Assign, PathAssignError
from jinja2 import Environment
from ruamel import yaml

from utils import (
    FLAGS,
    mkdir_p,
    ColoredFormatter,
    maxkeys,
    mapfunc,
    cached_property,
    flatten,
    walk
)

if os.path.exists("./.env"):
    load_dotenv()

logging.get_absl_handler().setFormatter(ColoredFormatter())
logger = logging.get_absl_logger()

RELEASE_TAG_FORMAT = "{release_type}-python{python_version}-{tag_suffix}"

DOCKERFILE_SUFFIX = ".Dockerfile.j2"
DOCKERFILE_HIERARCHY = ["base", "cudnn", "devel", "runtime"]
DOCKERFILE_NVIDIA_REGEX = re.compile(r'(?:nvidia|cuda|cudnn)+')
DOCKERFILE_NAME = DOCKERFILE_SUFFIX.split('.')[1]

SUPPORTED_PYTHON_VERSION = ["3.7", "3.8"]
SUPPORTED_OS = ["debian", "centos", "alpine", "amazonlinux"]

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"
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
      check_with: allowed_cudnn
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
    multistage_image:
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

    CUDNN_THRESHOLD = 1

    # TODO: custom check for releases.valuesrules

    def __init__(self, *args, **kwargs):
        if "packages" in kwargs:
            self.packages = kwargs["packages"]
        if "releases" in kwargs:
            self.releases = kwargs["releases"]
        self.cudnn_counter = 0
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

    def _check_with_allowed_cudnn(self, field, value):
        if 'cudnn' in value:
            self.cudnn_counter += 1
            pass
        if self.cudnn_counter > self.CUDNN_THRESHOLD:
            self._error(field, "Only allowed one CUDNN version per CUDA mapping")

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
                f"{registry}: Make sure to set {metadata['only_if']} or {metadata['user']}."
            )
        _user = os.getenv(metadata['user'])
        _pwd = os.getenv(metadata['pwd'])

        logger.info(f"Logging into Docker registry {registry} with credentials...")
        docker_client.login(username=_user, password=_pwd, registry=registry)
        repos[registry] = list(metadata['registry'].values())
    return repos


def upload_in_background(docker_client, image, registry, tag):
    """Upload a docker image (to be used by multiprocessing)."""
    image.tag(registry, tag=tag)
    logger.debug(f"Pushing {image} to {registry}...")
    for _line in docker_client.images.push(registry, tag=tag, stream=True, decode=True):
        print(json.dumps(_line, indent=2))


def load_manifest_yaml(file: str):
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

    return v.normalized(manifest)


def get_files(directory: pathlib.Path, pattern: str):
    for f in walk(directory):
        if pattern in f.name:
            return f


def get_data(obj: Union[Dict, MutableMapping], *path: str):
    """
    Get data from the object by dotted path.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        data = glom(obj, Path(*path))
    except PathAccessError:
        logger.exception(
            f"Exception occurred in get_data, unable to retrieve {'.'.join([*path])} from {repr(obj)}"
        )
    else:
        return data


def set_data(obj: Dict, value: Union[Dict, str], *path: str):
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(Path(*path), value))
    except ValueError:
        # we can just use dict.update to handle this error.
        obj.update(value)
    except PathAssignError:
        logger.exception(
            f"Exception occurred in update_data, unable to update {Path(*path)} with {value}"
        )


# noinspection PyUnresolvedReferences
class Generate(object):
    """Mixin for generating Dockerfile from templates."""

    _build_path = {}

    _template_env = Environment(
        extensions=["jinja2.ext.do"], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec: Dict, cuda_version: str):
        # set private class attributes to be manifest.yml keys
        for keys, values in release_spec.items():
            if keys == 'packages':
                mapfunc(lambda x: list(flatten(x)), values)
            super().__setattr__(f"_{keys}", values)

        # dirty hack to force add base
        for _v in self._packages.values():
            _v.update({'base': maxkeys(_v)})

        # setup cuda_version
        if cuda_version in self._cuda_dependencies.keys():
            self._cuda_version = cuda_version
        else:
            raise RuntimeError(f"{cuda_version} is not defined in manifest.yml")

    def __call__(self, dry_run=False, *args, **kwargs):
        if len(FLAGS.python_version) > 0:
            logger.debug("--python_version defined, ignoring SUPPORTED_PYTHON_VERSION")
            supported_python = FLAGS.python_version
            if supported_python not in SUPPORTED_PYTHON_VERSION:
                logger.critical(
                    f"{supported_python} is not supported. Allowed: {SUPPORTED_PYTHON_VERSION}"
                )
        else:
            supported_python = SUPPORTED_PYTHON_VERSION

        if not dry_run:
            # Consider generated directory ephemeral
            logger.info(f"Removing dockerfile dir: {FLAGS.dockerfile_dir}\n")
            shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
            mkdir_p(FLAGS.dockerfile_dir)

            return self.generate_registries(
                target_dir=FLAGS.dockerfile_dir, python_versions=supported_python
            )

    @cached_property
    def build_path(self):
        ordered = {
            k: self._build_path[k] for k in self._build_path.keys() if 'base' in k
        }
        ordered.update(self._build_path)
        return ordered

    @staticmethod
    def _generate_envars_context(bentoml_version: str):
        """
        Create a dictionary of args that will be parsed during runtime

        Args:
            bentoml_version: version of BentoML releases

        Returns:
            Dict of args
        """
        args = {"BENTOML_VERSION": bentoml_version}

        def update_args_dict(updater):
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
                logger.error(
                    f"cannot add to args dict with unknown type {type(updater)}"
                )
            return args

        return update_args_dict

    @staticmethod
    def _generate_tags_context(
        build_ctx: MutableMapping,
        python_version: str,
        release_type: str,
        fmt: Optional[str] = "",
    ):
        _core_fmt = fmt if fmt else RELEASE_TAG_FORMAT
        formatter = {
            "release_type": release_type,
            "tag_suffix": build_ctx["add_to_tags"],
            "python_version": python_version,
        }

        if release_type in ['runtime', 'cudnn']:
            set_data(
                formatter,
                get_data(build_ctx, 'envars', 'BENTOML_VERSION'),
                'release_type',
            )
            _tag = "-".join([_core_fmt.format(**formatter), release_type])
        else:
            _tag = _core_fmt.format(**formatter)

        # construct build_tag which is our base_image with $PYTHON_VERSION formats
        formatter.update({"python_version": "$PYTHON_VERSION", "release_type": "base"})
        _build_tag = _core_fmt.format(**formatter)

        return _tag, _build_tag

    @staticmethod
    def _generate_cudnn_context(cuda_components: Dict):
        cudnn_regex = re.compile(r"(cudnn)(\w+)")
        for k, v in cuda_components.items():
            if cudnn_regex.match(k):
                cudnn_release_version = cuda_components.pop(k)
                cudnn_major_version = cudnn_release_version.split(".")[0]
                return {
                    "version": cudnn_release_version,
                    "major_version": cudnn_major_version,
                }

    @staticmethod
    def _generate_distros_context(pack_def: MutableMapping):
        for rtype, dists in pack_def.items():
            for d in dists:
                yield d, rtype

    def generate_template_context(
        self,
        dist: str,
        dist_ver: str,
        dist_spec: Dict[str, str],
        py_ver: str,
        bento_ver: str,
        package: Optional[str] = None,
    ) -> Dict:
        """Generate template context for each distro releases."""
        _build_ctx = {}
        _cuda_components = get_data(self._cuda_dependencies, self._cuda_version).copy()

        if package:
            set_data(_build_ctx, package, 'package')
        set_data(_build_ctx, get_data(self._releases, dist, dist_ver))

        # setup envars
        set_data(
            _build_ctx,
            self._generate_envars_context(bento_ver)(dist_spec['envars']),
            'envars',
        )

        # check if python_version is supported.
        set_data(
            _build_ctx, py_ver, "envars", "PYTHON_VERSION",
        )

        # setup cuda deps
        if _build_ctx["cuda_prefix_url"]:
            major, minor, _ = self._cuda_version.rsplit(".")
            _build_ctx["cuda"] = {
                "ml_repo": NVIDIA_ML_REPO_URL.format(_build_ctx["cuda_prefix_url"]),
                "base_repo": NVIDIA_REPO_URL.format(_build_ctx["cuda_prefix_url"]),
                "version": {
                    "major": major,
                    "minor": minor,
                    "shortened": f"{major}.{minor}",
                },
                "cudnn": self._generate_cudnn_context(_cuda_components),
                "components": _cuda_components,
            }
            _build_ctx.pop('cuda_prefix_url')
        else:
            logger.warning(f"CUDA is disabled for {dist}. Skipping...")
            for key in list(_build_ctx.keys()):
                if DOCKERFILE_NVIDIA_REGEX.match(key):
                    _build_ctx.pop(key)

        return _build_ctx

    def render_from_templates(
        self,
        input_paths: List[pathlib.Path],
        output_path: pathlib.Path,
        metadata: Dict,
        build_tag: Optional[str] = None,
    ):
        for inp in input_paths:
            with inp.open('r') as inf:
                logger.debug(f"Processing template: {inp}")
                if DOCKERFILE_SUFFIX not in inp.name:
                    output_name = inp.stem
                else:
                    output_name = DOCKERFILE_NAME

                template = self._template_env.from_string(inf.read())
                if not output_path.exists():
                    output_path.mkdir(parents=True, exist_ok=True)

                with open(f"{output_path}/{output_name}", "w") as ouf:
                    ouf.write(template.render(metadata=metadata, build_tag=build_tag))
                ouf.close()
            inf.close()

    def generate_registries(self, target_dir: str, python_versions: List[str]):

        _tag_metadata = defaultdict()
        for combinations in itertools.product(self._packages.items(), python_versions):
            (package, pack_def), py_ver = combinations
            logger.info(f"Working on {package} for python{py_ver}")
            for (distro, rtype) in self._generate_distros_context(pack_def):
                for distro_version, distro_spec in self._releases[distro].items():
                    # setup our build context
                    _build_ctx = self.generate_template_context(
                        distro,
                        distro_version,
                        distro_spec,
                        py_ver,
                        FLAGS.bentoml_version,
                        package=package,
                    )
                    # generate our image tag.
                    _release_tag, _build_tag = self._generate_tags_context(
                        _build_ctx, py_ver, rtype,
                    )

                    template_dir = pathlib.Path(get_data(_build_ctx, 'templates_dir'))
                    base_output_path = pathlib.Path(
                        target_dir, package, f"{distro}{distro_version}",
                    )
                    if 'rhel' in template_dir.name and rtype == 'cudnn':
                        input_path = [
                            f
                            for f in template_dir.iterdir()
                            if DOCKERFILE_NVIDIA_REGEX.match(f.name)
                        ]
                    else:
                        input_path = [get_files(template_dir, str(rtype))]

                    if rtype != 'base':
                        output_path = pathlib.Path(base_output_path, rtype)
                    else:
                        output_path = base_output_path

                    if rtype == 'cudnn':
                        _metadata = {"docker_build_path": str(output_path)}
                        _metadata.update(_build_ctx)
                    else:
                        _metadata = _build_ctx

                    tag_keys = f"{_build_ctx['package']}:{_release_tag}"

                    # # setup directory correspondingly, and add these paths
                    output_dockerfile = str(pathlib.Path(output_path, DOCKERFILE_NAME))
                    set_data(_tag_metadata, _metadata, tag_keys)
                    set_data(self._build_path, output_dockerfile, tag_keys)
                    # generate our Dockerfile from templates.
                    self.render_from_templates(
                        input_paths=input_path,
                        output_path=output_path,
                        metadata=_metadata,
                        build_tag=_build_tag,
                    )
        return _tag_metadata


def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    print(f"\n{'-' * 59}\n| Configure Docker\n{'-' * 59}\n")
    # setup docker client
    docker_client = docker.from_env()

    # Parse specs from manifest.yml and validate it.
    release_spec = load_manifest_yaml(FLAGS.manifest_file)

    # Login Docker credentials
    repos = auth_registries(release_spec["repository"], docker_client)

    # generate templates to correct directory.
    print(f"\n{'-' * 59}\n| Generate from templates\n{'-' * 59}\n")
    generator = Generate(release_spec, cuda_version=FLAGS.cuda_version)
    tag_metadata = generator(dry_run=FLAGS.dry_run)
    build_path = generator.build_path

    if FLAGS.dump_metadata:
        logger.info(f"--dump_metadata is specified. Dumping metadata...")
        with open("./metadata.json", "w") as ouf:
            ouf.write(json.dumps(tag_metadata, indent=2))
        ouf.close()

    if FLAGS.stop_at_generate:
        logger.info(f"--stop_at_generate is specified. Stopping now...")
        return

    # We will build and push if args is parsed. Establish some variables.
    push_tags, logs = {}, []

    if not FLAGS.dry_run:
        print(f"\n{'-' * 59}\n| Building images\n{'-' * 59}\n")
        for image_tag, dockerfile_path in build_path.items():
            try:
                # this is should be when there are changed in base image.
                if FLAGS.overwrite:
                    docker_client.api.remove_image(image_tag, force=True)
                    # dirty workaround for not having goto supports in Python.
                    raise docker.errors.ImageNotFound(f"Building {image_tag}")
                elif docker_client.api.history(image_tag):
                    logger.info(
                        f"Image {image_tag} is already built and --overwrite is not specified. Skipping..."
                    )
                    push_tags[image_tag] = docker_client.images.get(image_tag)
                    continue
            except docker.errors.ImageNotFound:
                try:
                    logger.info(f"Building {image_tag} from {dockerfile_path}")
                    build_args = {
                        "PYTHON_VERSION": get_data(tag_metadata, image_tag, 'envars', 'PYTHON_VERSION')
                    }
                    # occasionally build process failed at install BentoML from PyPI. This could be due to
                    # us getting rate limited.
                    # https://warehouse.pypa.io/api-reference/index.html
                    resp = docker_client.api.build(
                        timeout=FLAGS.timeout,
                        path=".",
                        nocache=False,
                        buildargs=build_args,
                        dockerfile=dockerfile_path,
                        tag=image_tag,
                    )
                    last_event, image_id = None, None
                    while True:
                        try:
                            # output logs to stdout
                            # https://docker-py.readthedocs.io/en/stable/user_guides/multiplex.html
                            output = next(resp).decode('utf-8')
                            json_output = json.loads(output.strip('\r\n'))
                            if 'stream' in json_output:
                                # output docker build process
                                print(json_output['stream'])
                                built = re.compile(
                                    r'(^Successfully built |sha256:)([0-9a-f]+)$'
                                )
                                matched = built.search(json_output['stream'])
                                if matched:
                                    image_id = matched.group(2)
                                last_event = json_output['stream']
                                logs.append(json_output)
                        except StopIteration:
                            logger.debug(f"Successfully built {image_tag}.")
                            break
                        except ValueError:
                            logger.error(f"Errors while building image:\n{output}")
                    if image_id:
                        push_tags[image_tag] = docker_client.images.get(image_id)
                    else:
                        raise docker.errors.BuildError(last_event or 'Unknown', logs)
                except docker.errors.BuildError as e:
                    logger.error(f"Failed to build {image_tag} :\n{e.msg}")
                    for line in e.build_log:
                        if 'stream' in line:
                            print(line['stream'].strip())
                    logger.fatal('ABORTING due to failure!')

    # Push process
    # whether to push base image or not
    if FLAGS.push_to_hub:
        print(f"\n{'-' * 59}\n| Pushing images\n{'-' * 59}\n")
        assert push_tags, logger.error("Push tags is empty!!")
        for repo, registries in repos.items():
            for image_tag, image in push_tags.items():
                _registry, tag = image_tag.split(":")
                registry = ''.join((k for k in registries if _registry in k))
                logger.info(f"Uploading {image_tag} to {repo}")
                if not FLAGS.dry_run:
                    p = multiprocessing.Process(
                        target=upload_in_background,
                        args=(docker_client, image, registry, tag),
                    )
                    p.start()
                    # upload_in_background(docker_client, image, registry, tag)
    else:
        logger.info("--push_to_hub is not specified. Skip pushing images...")


if __name__ == "__main__":
    app.run(main)