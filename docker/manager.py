#!/usr/bin/env python3
import json
import logging.config
import pathlib
from collections.abc import Iterable
from decimal import Decimal
from typing import Dict, Optional

from absl import flags, app
from cerberus import Validator, TypeDefinition
from glom import glom, Path, Assign, PathAccessError, PathAssignError
from jinja2 import Environment
from ruamel import yaml

logger = logging.getLogger(__name__)
# defined global vars.
FLAGS = flags.FLAGS

DOCKERFILE_PREFIX = ".Dockerfile.j2"

SUPPORTED_PYTHON_VERSION = ["3.6", "3.7", "3.8"]
SUPPORTED_OS = ['ubuntu', 'debian', 'centos', 'alpine', 'amazonlinux']

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"

# cerberus
DECIMAL_TYPE = TypeDefinition('decimal', (Decimal, str), ())

flags.DEFINE_boolean(
    "push_to_hub",
    False,
    "Whether to upload images to given registries.",
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
flags.DEFINE_string(
    "templates_dir", "./templates", "templates directory", short_name="t"
)
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file", short_name="m")

flags.DEFINE_string("bentoml_version", None, "BentoML release version", required=True)

flags.DEFINE_string(
    "cuda_version",
    None,
    "Define CUDA version",
    short_name='c',
    required=True
)

flags.DEFINE_string("python_version", "3.8", "Define a Python version to build Docker images.")

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
    requires:
      type: string
      dependencies: cuda
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
    types_mapping = Validator.types_mapping.copy()
    types_mapping['decimal'] = DECIMAL_TYPE

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
    v = MetadataSpecValidator(v_schema, packages=manifest['packages'], releases=manifest['releases'])

    if not v.validate(manifest):
        logger.error(f"{file} is invalid. Errors as follow:")
        logger.error(yaml.dump(v.errors, indent=2))
        exit(1)

    release_spec = v.normalized(manifest)
    return release_spec


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
        logger.error(f"Exception occurred in get_data, unable to retrieve from {' '.join(*path)}")
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
        logger.error(
            f"Exception occurred in update_data, unable to update {Path(*path)} with {value}")


def update_args_dict(args_dict, updater):
    # Args will have format ARG=foobar
    if isinstance(updater, str):
        args, _, value = updater.partition("=")
        args_dict[args] = value
        return
    elif isinstance(updater, list):
        for v in updater:
            update_args_dict(args_dict, v)
    elif isinstance(updater, dict):
        for args, value in updater.items():
            args_dict[args] = value
    else:
        raise AttributeError(f"cannot add to args dict with unknown type {type(updater)}")


def build_release_args(distro_release_spec, bentoml_version):
    """
    Create a dictionary of args that will be parsed during runtime

    Args:
        distro_release_spec: dict that follow release_spec templates defined in manifest.yml
        bentoml_version: version of BentoML releases

    Returns:
        Dict of args
    """
    args = {}
    update_args_dict(args, distro_release_spec['envars'])
    # we will update PYTHON_VERSION when build process is invoked
    args['PYTHON_VERSION'] = ''
    args['BENTOML_VERSION'] = bentoml_version
    return args


class TemplatesMixin(object):
    """
    Mixin for generating Dockerfile from templates.
    """

    _build_ctx = {}

    template_env = Environment(extensions=['jinja2.ext.do'], trim_blocks=True, lstrip_blocks=True)

    def __init__(self, release_spec):
        self._metadata = release_spec
        print(release_spec.keys())
        # setup default keys for build context
        self._add_flatten_ctx(release_spec['release_spec'], 'cuda')

        # setup cuda_version and cuda_components
        if self._check_cuda_version(FLAGS.cuda_version):
            self.cuda_version = FLAGS.cuda_version
        else:
            raise RuntimeError(f"CUDA {FLAGS.cuda_version} is either not supported or defined under manifest.cuda")
        self.cuda_components = get_data(self._metadata, 'cuda_dependencies', self.cuda_version)

    def __call__(self, *args, **kwargs):
        distro = 'ubuntu'
        distro_version = 20.04
        distro_spec = self._metadata['releases'][distro][distro_version]
        self.prepare_template_context(distro, distro_version, distro_spec)
        print(json.dumps(self._build_ctx, indent=2))
        print()

        self.generate_dockerfiles()

    def _check_cuda_version(self, cuda_version, cuda_path: Optional[str] = 'cuda_dependencies'):
        # get defined cuda version for cuda
        return True if cuda_version in self._metadata[cuda_path].keys() else False

    def _get_cudnn_version(self):
        cudnn = []
        for k, v in self.cuda_components.items():
            if k.startswith('cudnn') and v:
                cudnn.append(k)
        return cudnn

    def _add_flatten_ctx(self, spec_dict: Dict, *ignore_keys):
        # flatten release_spec to template context.
        for k, v in spec_dict.items():
            if k in list(flatten([*ignore_keys])):
                continue
            self._build_ctx[k] = v

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

        self._add_flatten_ctx(get_data(self._metadata, 'releases', distro, distro_version), 'cuda')

        self._build_ctx['envars'] = build_release_args(distro_release_spec, FLAGS.bentoml_version)

        # setup cuda deps
        if self._build_ctx['cuda_prefix_url'] != "":
            major, minor, _ = self.cuda_version.rsplit(".")
            self._build_ctx['cuda'] = {
                'ml_repo': NVIDIA_ML_REPO_URL.format(self._build_ctx['cuda_prefix_url']),
                'base_repo': NVIDIA_REPO_URL.format(self._build_ctx['cuda_prefix_url']),
                'version': {
                    'release': self.cuda_version,
                    'major': major,
                    'minor': minor,
                    'shortened': ".".join([major, minor]),
                },
                'components': self.cuda_components
            }
        else:
            logger.debug("distro doesn't enable CUDA. Skipping...")
            self._build_ctx['cuda'] = {}

    def output_template(self, input_tmpl: pathlib.PosixPath, output_path, ctx=None):
        ctx = ctx if ctx is not None else self._build_ctx
        with open(input_tmpl, 'r') as inf:
            logger.debug(f"Processing template: {input_tmpl}")
            full_output_path = pathlib.Path(output_path)
            if DOCKERFILE_PREFIX not in input_tmpl.name:
                logger.debug(f"Processing {input_tmpl.name}")
                output_name = input_tmpl[:-len(DOCKERFILE_PREFIX.split('.')[1])]
            else:
                output_name = DOCKERFILE_PREFIX.split('.')[0]

            template = self.template_env.from_string(inf.read())
            if not full_output_path.exists():
                logger.debug(f"Creating {full_output_path}")
                full_output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{full_output_path}/{output_name}", 'w') as ouf:
                ouf.write(template.render(metadata=ctx))
            ouf.close()
        inf.close()

    def _get_release_package_supported_distro(self):
        print(get_data(self._metadata, 'packages'))

    def generate_dockerfiles(self):
        self._get_release_package_supported_distro()


def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    # Parse specs from manifest.yml and validate it.
    release_spec = load_manifest_yaml(FLAGS.manifest_file)
    # print(yaml.dump(release_spec, indent=2, explicit_start=True))

    tmpl = TemplatesMixin(release_spec)
    tmpl()

    # Setup Docker credentials
    # this includes push directory, envars


if __name__ == "__main__":
    app.run(main)