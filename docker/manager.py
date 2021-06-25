#!/usr/bin/env python3
import errno
import json
import logging
import os
import pathlib
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Dict, Optional

from absl import flags, app
from cerberus import Validator
from glom import glom, Path, PathAccessError
from jinja2 import Environment
from ruamel import yaml

# defined global vars.
FLAGS = flags.FLAGS

DOCKERFILE_PREFIX = ".Dockerfile.j2"
DOCKERFILE_ALLOWED_NAMES = ['base', 'cudnn', 'devel', 'runtime']

SUPPORTED_PYTHON_VERSION = ["3.6", "3.7", "3.8"]
SUPPORTED_OS = ['ubuntu', 'debian', 'centos', 'alpine', 'amazonlinux']

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"
)

flags.DEFINE_boolean(
    "push_to_hub", False, "Whether to upload images to given registries.",
)

# CLI-related
flags.DEFINE_boolean(
    "dump_metadata", False, "Whether to dump all tags metadata to file."
)
flags.DEFINE_boolean(
    "dry_run", False, "Whether to dry run. This won't create Dockerfile."
)
flags.DEFINE_boolean(
    "generate_dockerfile", False, "Whether to just generate dockerfile."
)
flags.DEFINE_boolean("build_images", False, "Whether to build images.")
flags.DEFINE_boolean(
    "stop_on_failure",
    False,
    "Stop processing tags if any one build fails. If False or not specified, failures are reported but do not affect "
    "the other images.",
    short_name="s",
)
# directory and files
flags.DEFINE_string(
    "dockerfile_dir",
    "./generated",
    "path to generated Dockerfile. Existing files will be deleted with new Dockerfiles",
    short_name="g",
)
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file", short_name="m")

flags.DEFINE_string("bentoml_version", None, "BentoML release version", required=True)

flags.DEFINE_string("cuda_version", "11.3.1", "Define CUDA version", short_name='c')

flags.DEFINE_string("python_version", "3.8", "Python version to build Docker images.")

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


def validate_dockerfile_name(name: str):
    """Validate our input file name for Dockerfile templates."""
    if DOCKERFILE_PREFIX in name:
        name, _, _ = name.partition(DOCKERFILE_PREFIX)
        if name not in DOCKERFILE_ALLOWED_NAMES:
            raise RuntimeError(
                f"Unable to parse dockerfile: {name}. Dockerfile directory should only consists of {DOCKERFILE_ALLOWED_NAMES}"
            )


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
        logging.error(
            f"Exception occurred in get_data, unable to retrieve from {' '.join(*path)}"
        )
    else:
        return data


def build_release_args(bentoml_version):
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
            raise AttributeError(
                f"cannot add to args dict with unknown type {type(updater)}"
            )
        return args

    return update_args_dict


def flatten(arr):
    for it in arr:
        if isinstance(it, Iterable) and not isinstance(it, str):
            for x in flatten(it):
                yield x
        else:
            yield it


def load_manifest_yaml(file):
    with open(file, 'r') as input_file:
        manifest = yaml.safe_load(input_file)

    v_schema = yaml.safe_load(SPEC_SCHEMA)
    v = MetadataSpecValidator(
        v_schema, packages=manifest['packages'], releases=manifest['releases']
    )

    if not v.validate(manifest):
        logging.error(f"{file} is invalid. Errors as follow:\n")
        logging.error(yaml.dump(v.errors, indent=2))
        exit(1)

    release_spec = v.normalized(manifest)
    return release_spec


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
        if 'packages' in kwargs:
            self.packages = kwargs['packages']
        if 'releases' in kwargs:
            self.releases = kwargs['releases']
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


# noinspection PyPep8Naming
class cached_property(property):
    """Direct ports from bentoml/utils/__init__.py"""

    class _Missing(object):
        def __repr__(self):
            return "no value"

        def __reduce__(self):
            return "_missing"

    _missing = _Missing()

    def __init__(
            self, func, name=None, doc=None
    ):  # pylint:disable=super-init-not-called
        super().__init__(doc)
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, types=None):  # pylint:disable=redefined-builtin
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
    _build_path = {}

    _template_env = Environment(
        extensions=['jinja2.ext.do'], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec):
        # set private class attributes to be manifest.yml keys
        for keys, values in release_spec.items():
            super().__setattr__(f"_{keys}", values)

        # setup default keys for build context
        self._update_build_ctx(self._release_spec, 'cuda')

        # setup cuda_version and cuda_components
        if self._check_cuda_version(FLAGS.cuda_version):
            self._cuda_version = FLAGS.cuda_version
        else:
            raise RuntimeError(
                f"CUDA {FLAGS.cuda_version} is either not supported or defined in {self._cuda_dependencies}"
            )
        self._cuda_components = get_data(self._cuda_dependencies, self._cuda_version)

    def __call__(self, *args, **kwargs):
        self.generate_dockerfiles()

    @cached_property
    def build_path(self):
        return self._build_path

    def _check_cuda_version(self, cuda_version):
        # get defined cuda version for cuda
        return True if cuda_version in self._cuda_dependencies.keys() else False

    def _update_build_ctx(self, spec_dict: Dict, *ignore_keys):
        # flatten release_spec to template context.
        for k, v in spec_dict.items():
            if k in list(flatten([*ignore_keys])):
                continue
            self._build_ctx[k] = v

    def _get_distro_for_supported_package(self, package):
        """Return a dict containing keys as distro and values as release type (devel, cudnn, runtime)"""
        releases = defaultdict(list)
        for k, v in self._packages[package].items():
            _v = list(flatten(v))
            for distro in _v:
                releases[distro].append(k)
        return releases

    def prepare_tags(self, distro, distro_version):
        tags = ""
        return tags

    def prepare_template_context(self, distro, distro_version, distro_release_spec):
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
        cuda_regex = re.compile(r"cuda*")

        self._update_build_ctx(get_data(self._releases, distro, distro_version), 'cuda')

        self._build_ctx['envars'] = build_release_args(FLAGS.bentoml_version)(
            distro_release_spec['envars']
        )

        if self._build_ctx['needs_deps_image'] is True:
            self._build_ctx['base_tags'] = 'base'
            self._build_ctx.pop('needs_deps_image')
        else:
            self._build_ctx.pop('base_tags')

        # setup cuda deps
        if self._build_ctx['cuda_prefix_url'] not in ["", None]:
            major, minor, _ = self._cuda_version.rsplit(".")
            self._build_ctx['cuda'] = {
                'ml_repo': NVIDIA_ML_REPO_URL.format(
                    self._build_ctx['cuda_prefix_url']
                ),
                'base_repo': NVIDIA_REPO_URL.format(self._build_ctx['cuda_prefix_url']),
                'version': {
                    'release': self._cuda_version,
                    'major': major,
                    'minor': minor,
                    'shortened': ".".join([major, minor]),
                },
                'components': self._cuda_components,
            }
            self._build_ctx.pop('cuda_prefix_url')
        else:
            logging.debug("distro doesn't enable CUDA. Skipping...")
            for key in list(self._build_ctx.keys()):
                if cuda_regex.match(key):
                    self._build_ctx.pop(key)

    def render_dockerfile_from_templates(
            self,
            input_tmpl_path: pathlib.Path,
            build_path: pathlib.Path,
            ctx: Optional[Dict] = None,
    ):
        ctx = ctx if ctx is not None else self._build_ctx
        with open(input_tmpl_path, 'r') as inf:
            logging.debug(f"Processing template: {input_tmpl_path}")
            if DOCKERFILE_PREFIX not in input_tmpl_path.name:
                logging.debug(f"Processing {input_tmpl_path.name}")
                output_name = input_tmpl_path[: -len(DOCKERFILE_PREFIX.split('.')[2])]
            else:
                output_name = DOCKERFILE_PREFIX.split('.')[1]

            template = self._template_env.from_string(inf.read())
            if not build_path.exists():
                logging.debug(f"Creating {build_path}")
                build_path.mkdir(parents=True, exist_ok=True)

            with open(f"{build_path}/{output_name}", 'w') as ouf:
                ouf.write(template.render(metadata=ctx))
            ouf.close()
        inf.close()

    def generate_dockerfiles(self):
        """
        Workflow:
        walk all templates dir from given context

        our targeted generate directory should look like:
        generated
            -- model-server
                -- ubuntu
                    -- 20.04
                        -- runtime
                            -- Dockerfile
                        -- devel
                            -- Dockerfile
                        -- cudnn
                            -- Dockerfile
                        -- Dockerfile (for base deps images)
                    -- 18.04
                -- centos
            -- yatai-service
        """
        for package in self._packages.keys():
            # update our packages ctx.
            self._build_ctx['packages'] = package
            _distros_release_type = self._get_distro_for_supported_package(package)
            for distro, release_type in _distros_release_type.items():
                for distro_version, distro_spec in self._releases[distro].items():
                    logging.info(f"Processing: {distro}{distro_version} for {package}")
                    # setup our build context
                    self.prepare_template_context(distro, distro_version, distro_spec)
                    logging.info("\n" + json.dumps(self._build_ctx, indent=2))

                    for root, _, files in os.walk(self._build_ctx['templates_dir']):
                        for f in files:
                            validate_dockerfile_name(f)

                    # setup directory correspondingly, and add these paths
                    # with correct tags name under self._build_path

        # self.output_template(
        #     input_tmpl_path=pathlib.Path(
        #         pathlib.Path(
        #             self._build_ctx['templates_dir'], f'base{DOCKERFILE_PREFIX}'
        #         )
        #     ),
        #     build_path=pathlib.Path(
        #         f"{FLAGS.dockerfile_dir}/model-server/ubuntu/20.04"
        #     ),
        # )


def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    # Parse specs from manifest.yml and validate it.
    release_spec = load_manifest_yaml(FLAGS.manifest_file)

    tmpl_mixin = TemplatesMixin(release_spec)
    tmpl_mixin()

    if FLAGS.generate_dockerfile:
        return

    # Setup Docker credentials
    # this includes push directory, envars


if __name__ == "__main__":
    app.run(main)