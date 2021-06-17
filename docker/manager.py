#!/usr/bin/env python3

# workflow
#   parse manifest.yml
#   get content from partials, and check if the file is correctly format
#   assemble tags and images given from manifest.yml
#   Empty our generated directory and generate new Dockerfile
#
import os

from collections import defaultdict
from absl import flags, app
from cerberus import Validator
import yaml
import sys
import docker

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"
)

SPEC_SCHEMA = """
header: 
  type: string

releases:
  type: dict
  keysrules:
    type: string
  valuesrules:
    type: dict
    schema:
      spec:
        type: list
        required: true
        schema: 
          type: string

dist:
  type: dict
  keysrules:
    type: string
  valuesrules:
    type: list
    schema:
      type: dict
      schema:
        add_to_name:
          type: string
        overwrite_dockerfile_name: 
          type: string
        args:
          type: list
          default: []
          schema:
            type: string
            isargs: true
        partials:
          type: list
          schema:
            type: string
            ispartial: true

"""

FLAGS = flags.FLAGS

flags.DEFINE_string('partial_dir', "./partials", 'partials directory')
flags.DEFINE_string('manifest_file', "./manifest.yml", 'manifest file')

flags.DEFINE_string('bentoml_version', "0.13.0", 'BentoML version')


class DockerTagValidator(Validator):
    """
    Custom cerberus validator for BentoML docker release spec.
    
    Refers to https://docs.python-cerberus.org/en/stable/customize.html#custom-rules.[
    This will add two rules:
    - args should have the correct format: ARG=foobar
    - partial file that construct docker images should have correct format: *.partial.Dockerfile

    Args:
        partials: directory containing all partial files
    """

    def __init__(self, *args, **kwargs):
        if 'partials' in kwargs:
            self.partials = kwargs['partials']
        super(Validator, self).__init__(*args, **kwargs)

    def _validate_ispartial(self, ispartial, field, value):
        """
        Validate the partial references of a partial spec

        Args:
            ispartial: Value of the rule, bool
            field: The field being validated
            value: The field's value

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if ispartial and value not in self.partials:
            self._error(field, f"{value} is not present in partials directory")

    def _validate_isargs(self, isargs, field, value):
        """
        Validate a string is either ARG=foobar ARG foobar.

        Args:
            isargs: Value of the rule, bool
            field: The field being validated
            value: The field's value

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if isargs and '=' not in value:
            self._error(
                field, f"{value} should have form ARG=foobar since isargs={isargs}"
            )
        if not isargs and '=' in value:
            self._error(
                field, f"{value} should have form ARG foobar since isargs={isargs}"
            )


def assemble_tag(release_spec, release_version, all_partials):
    """
    Assemble all tags based on given spec.

    Args:
        release_spec: Nested dict containing tag spec
        release_version: Bentoml release version
        all_partials: Dict containing every partials with its content.

    Returns:
        Dict of tags and content of Dockerfile.
    """
    result_data = defaultdict(list)
    for package, nametype in release_spec['releases'].items():
        print(package)
        print(nametype)

    return result_data


def get_partials_content(partial_path):
    """
    Get partials name and its content
    Args:
        partial_path: directory containing partial

    Returns:
        {partial_file_name: partial_file_content} where partial_file_name have format: foo/bar
    """
    partials = {}
    for path, _, file in os.walk(partial_path):
        for name in file:
            full_path = os.path.join(path, name)
            if '.partial.Dockerfile' not in full_path:
                print(f"skipping {full_path} since it is not a partial")
                continue
            # partial/foo/bar.partial.Dockerfile -> foo/bar
            _simple = full_path[len(partial_path) + 1: -len('.partial.Dockerfile')]
            with open(full_path, 'r') as f:
                contents = f.read()
            partials[_simple] = contents
    return partials


def generate_dockerfile_from_partial(headers, used_partials, all_partials):
    """Merge required partials with headers"""
    used_partials = list(used_partials)
    return '\n'.join([headers] + [all_partials[u] for u in used_partials])


def main(argv):
    if len(argv) > 1:
        app.UsageError('Too many command line args.')

    # parse manifest.yml
    schema = yaml.safe_load(SPEC_SCHEMA)
    with open(FLAGS.manifest_file, 'r') as manifest_file:
        spec = yaml.safe_load(manifest_file)

    # validate manifest configs and parse partial contents and spec
    partials = get_partials_content(FLAGS.partial_dir)
    v = DockerTagValidator(schema, partials=partials)
    if not v.validate(spec):
        print(f"ERROR: {FLAGS.manifest_file} is invalid.")
        print(yaml.dump(v.errors, indent=2))
        exit(1)
    release_spec = v.normalized(spec)

    # assemble tags and dockerfile used to build required images.
    all_tags = assemble_tag(release_spec, FLAGS.bentoml_version, partials)


if __name__ == '__main__':
    app.run(main)