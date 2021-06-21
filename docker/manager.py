#!/usr/bin/env python3
import inspect
import logging
import sys
from collections.abc import Iterable

from absl import flags, app
from cerberus import Validator
from ruamel import yaml

# defined global vars.
# logging.basicConfig(level=logging.DEBUG,
#                     format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
#                     datefmt="%d/%b/%Y %H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS
SUPPORTED_OS = ['ubuntu', 'slim', 'centos', 'alpine', 'amazonlinux']

# CLI Flags
# Docker-related
flags.DEFINE_string(
    "hub_password",
    None,
    "Docker Hub password, only used with --upload_to_hub. Use from env params or pipe it in to avoid saving it to "
    "history ",
)
flags.DEFINE_string(
    "hub_username", None, "Docker Hub username, only used with --upload_to_hub"
)
flags.DEFINE_string(
    "hub_org", "bentoml", "Docker Hub repository, only used with --upload_to_hub"
)
flags.DEFINE_boolean(
    "upload_to_hub",
    False,
    "Whether to upload images to docker.io, you should also provide --hub_password and "
    "--hub_username ",
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
    "templates_dir", "./templates", "templates directory", short_name="d"
)
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file", short_name="m")

flags.DEFINE_multi_string(
    "supported_python_version",
    ["3.6", "3.7", "3.8"],
    "Supported Python version",
    short_name="p",
)
flags.DEFINE_string("bentoml_version", None, "BentoML release version")

SPEC_SCHEMA = """
--- 
header: 
  type: string
  
repository: 
  type: dict
  keysrules: 
    type: string
  valuesrules: 
    type: dict
    schema: 
      only_if: 
        env_vars: true
        type: string
      pwd: 
        env_vars: true
        type: string
      registry: 
        keysrules: 
          check_with: packages
          type: string
        type: dict
        valuesrules: 
          type: string
      user: 
        env_vars: true
        type: string
        
packages: 
  type: dict
  keysrules: 
    type: string
  valuesrules: 
    type: dict
    allowed: [devel, cudnn, runtime]
    schema: 
      devel: 
        type: list
        schema: 
          type: string
        supported_dists: true
      cudnn: 
        type: [string, list]
        supported_dists: true
      runtime: 
        required: true
        type: [string, list]
        supported_dists: true
  
dependencies: 
  type: dict
  schema:
    cuda: 
      type: dict
      keysrules: 
        type: string
        regex: '(v\d{1,2}\.\d{1}\.\d)'
      valuesrules: 
        type: dict
        keysrules: 
          type: string
        valuesrules: 
          type: dict
          schema:
            version:
              type: string
    
release_spec: 
  required: true
  type: dict
  schema: 
    templates_dir: 
      type: string
    envars:
      type: list
      default: []
      schema:
        check_with: args_format
        type: string
    base_image:
      type: string
    add_to_tags:
      type: string
      required: true
    cuda:
      type: dict
        
releases: 
  type: dict
  keysrules: 
    supported_dists: true
    type: string
  valuesrules: 
    type: dict
        
"""


def flatten(arr):
    for it in arr:
        if isinstance(it, Iterable) and not isinstance(it, str):
            for x in flatten(it):
                yield x
        else:
            yield it


class DockerTagValidator(Validator):
    """
    Custom cerberus validator for BentoML docker release spec.

    Refers to https://docs.python-cerberus.org/en/stable/customize.html#custom-rules.
    This will add two rules:
    - args should have the correct format: ARG=foobar as well as correct ENV format.
    - releases should be defined under SUPPORTED_OS

    Args:
        packages: bentoml release packages, model-server, yatai-service, etc
    """

    def __init__(self, *args, **kwargs):
        if 'packages' in kwargs:
            self.packages = kwargs['packages']
        super(Validator, self).__init__(*args, **kwargs)

    def _check_with_packages(self, field, value):
        """
        Check if registry is defined in packages.
        """
        if value not in self.packages:
            self._error(field, f"{field} is not defined under packages")

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


# TODO: custom check for releases.valuesrules
def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")
    try:
        with open(FLAGS.manifest_file, "r") as input_file:
            manifest = yaml.safe_load(input_file)

        v_schema = yaml.safe_load(SPEC_SCHEMA)
        v = DockerTagValidator(v_schema, packages=manifest['packages'].keys())

        if not v.validate(manifest):
            logger.error(f"{FLAGS.manifest_file} is invalid. Errors as follow:")
            print(yaml.dump(v.errors, indent=2))
            exit(1)
        release_spec = v.normalized(manifest)

        print(yaml.dump(release_spec, indent=2))
    except KeyboardInterrupt:
        logger.error("Interrupt. Cleaning cache file...")


if __name__ == "__main__":
    app.run(main)
