#!/usr/bin/env python3

# workflow
#   parse manifest.yml
#   get content from partials, and check if the file is correctly format
#   assemble tags and images given from manifest.yml
#   Empty our generated directory and generate new Dockerfile
#
# naming covention: ${release_version | devel}-${python_version}-${os}-cuda${cuda_version}
#   where optional_args can be:
#     - cudnn${cudnn_major_version} includes cudart and cudnn
#     - cuda${cuda_version} that only contain cudart and cublas
#   where os can be:
#     - ubuntu${ubuntu_version}
#     - alpine${alpine_version}
#     - slim
# example: 0.13.1-python3.8-ubuntu20.04-gpu
#          0.13.1-python3.8-alpine3.14
#          0.13.1-python3.7-slim
#          0.13.1-python3.7-slim-gpu
#          0.13.1-python3.6-amazonlinux2
# NOTE: cuda is only supported for slim, ubuntu, and centos
#
# content structure of manifest.yml (only for my sanity)
#
#     release: dict
#       key: str -> type of image {devel/release}
#       value: list
#         value: defined spec under dist
#         -> this will allow us to how to build tags and ensemble our images from partial files
#         - for now we are introducing spec system based on supported OS and its coresponding GPU support
#
#     dist: dict
#       key: str -> os bentoml going to support
#       value: dict -> contains options to parse directly back to release
#         add_to_name: str -> handle tag naming
#         overwrite_dockerfile_name: str -> overwrite our generated dockerfile for this given image
#         args: list[str] this will be parsed at docker buildtime
#         partials: list[str] components needed to build our dockerfile


import itertools
import logging
import os
import re
from collections import defaultdict
from copy import deepcopy

import yaml
from absl import flags, app
from cerberus import Validator

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

flags.DEFINE_string("generated", "./generated",
                    "path to generated Dockerfile. Existing files will be deleted with new Dockerfiles")
flags.DEFINE_string("partial_dir", "./partials", "Partials directory")
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file")

flags.DEFINE_multi_string("supported_python_version", ["3.6", "3.7", "3.8"], "Supported Python version", short_name='p')
flags.DEFINE_string("bentoml_version", None, "BentoML release version", required=True)

SPEC_SCHEMA = """
header: 
  type: string

releases:
  type: dict
  keysrules:
    type: string
  valuesrules:
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
        
        
        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if ispartial and value not in self.partials:
            self._error(field, f"{value} is not present in partials directory")

    def _validate_isargs(self, isargs, field, value):
        """
        Validate a string is either ARG=foobar ARG foobar.

        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if isargs and "=" not in value:
            self._error(
                field, f"{value} should have format ARG=foobar since isargs={isargs}"
            )
        if not isargs and "=" in value:
            self._error(
                field, f"{value} should have format ARG foobar since isargs={isargs}"
            )


def aggregate_all_combinations_from_dist_spec(release_spec, used_dist):
    """Get all possible groupings for a tag spec."""
    dist_sets = deepcopy(release_spec['dist'])
    for name in used_dist:
        for d in dist_sets[name]:
            d['set_name'] = name

    grouped_but_not_keyed = [dist_sets[name] for name in used_dist]
    return list(itertools.product(*grouped_but_not_keyed))


def build_release_tags(string, dists, bento_version, python_version):
    """
    Build tags name from a list of dist.

    Args:
        string: Dist_spec {foo} to format with set_name
        dists: List of dist
        bento_version: BentoML version
        python_version: Supported python version

    Returns:
        Release tag of images
    """
    formatter = {}
    formatter.update({d['set_name']: d['add_to_name'] for d in dists})
    formatter.update({d['set_name']: d['overwrite_dockerfile_name'] for d in dists if 'overwrite_dockerfile_name' in d})
    name = string.format(**formatter)

    if "devel" in name:
        _, dev, rest = name.partition("devel")
        return "-".join([dev, f"python{python_version}", rest[1:]])
    return "-".join([bento_version, f"python{python_version}", name])


def get_dist_spec(dist_specs, tag_spec):
    """
    Extract dist spec and args for that dist string:
    {foo}{bar} will find foo and bar, assuming foo and bar are named dist sets.

    Args:
        dist_specs: Dict of named dist sets
        tag_spec: spec string {foo}{bar}

    Returns:
        used_dist
    """
    used_dist = []
    loaded_bracket = re.compile(r"{([^{}]+)}")
    possible_dist_and_args = loaded_bracket.findall(tag_spec)
    for name in possible_dist_and_args:
        if name in dist_specs:
            used_dist.append(name)
        else:
            logger.warning(f"WARNING: {name} is not yet defined under dist")

    return used_dist


def build_dist_args(dists, ver):
    """Build dictionary of required args for each dist."""

    def update_args_dict(args_dict, updater):
        """Update dict of args from a list or a dict."""
        if isinstance(updater, list):
            for u in updater:
                key, sep, value = u.partition("=")
                if sep == "=":
                    args_dict[key] = value
        if isinstance(updater, dict):
            for key, value in updater.items():
                args_dict[key] = value
        return args_dict

    args = {}
    for d in dists:
        args = update_args_dict(args, d['args'])
    # creates args for python version.
    args["PYTHON_VERSION"] = ver
    return args


def get_dist_items(dists, key):
    """Get flattened list of certain key in list of dist."""
    return list(itertools.chain(*[d[key] for d in dists if key in d]))


def assemble_tag(release_spec, bento_ver, supported_py_ver, all_partials):
    """
    Assemble all tags based on given spec.

    Args:
        release_spec: Nested dict containing tag spec
        bento_ver: Bentoml release version
        supported_py_ver: Supported python release version
        all_partials: Dict containing every partials with its content.

    Returns:
        Dict of tags and content of Dockerfile.
    """
    tag_metadata = defaultdict(list)
    for package, release in release_spec["releases"].items():
        # package: model-server or yatai-service -> release: devel, versioned
        for release_type, distro in release.items():
            for tag_spec in distro["spec"]:
                used_dist = get_dist_spec(release_spec["dist"], tag_spec)

                dist_combos = aggregate_all_combinations_from_dist_spec(release_spec, used_dist)

                for dists in dist_combos:
                    for pyver in supported_py_ver:
                        tag_args = build_dist_args(dists, pyver)
                        tag_name = build_release_tags(tag_spec, dists, bento_ver, pyver)
                        used_partials = get_dist_items(dists, 'partials')
                        dockerfile_contents = merge_partials(release_spec["header"], used_partials, all_partials)

                        tag_metadata[tag_name].append({
                            'package': package,
                            'release': release_type,
                            'tag_spec': tag_spec,
                            'docker_args': tag_args,
                            'partials': used_partials,
                            'dockerfile_contents': dockerfile_contents,
                        })

    return tag_metadata


def get_partials_content(partial_path):
    """
    Get partials name and its content
    Args:
        partial_path: directory containing partial

    Returns:
        Dict of partial simple call and its content
    """
    partials = {}
    for path, _, file in os.walk(partial_path):
        for name in file:
            full_path = os.path.join(path, name)
            if ".partial.dockerfile" not in full_path:
                logger.debug(f"skipping {full_path} since it is not a partial")
                continue
            # partial/foo/bar.partial.dockerfile -> foo/bar
            _simple = full_path[len(partial_path) + 1: -len(".partial.dockerfile")]
            with open(full_path, "r") as f:
                contents = f.read()
            partials[_simple] = contents
    return partials


def merge_partials(headers, used_partials, all_partials):
    """Merge required partials with headers to generate Dockerfile"""
    used_partials = list(used_partials)
    return "\n".join([headers] + [all_partials[u] for u in used_partials])


def main(argv):
    if len(argv) > 1:
        app.UsageError("Too many command line args.")

    # parse manifest.yml
    schema = yaml.safe_load(SPEC_SCHEMA)
    with open(FLAGS.manifest_file, "r") as manifest_file:
        spec = yaml.safe_load(manifest_file)

    # validate manifest configs and parse partial contents and spec
    partials = get_partials_content(FLAGS.partial_dir)
    v = DockerTagValidator(schema, partials=partials)
    if not v.validate(spec):
        logger.error(f"{FLAGS.manifest_file} is invalid.")
        logger.error(yaml.dump(v.errors, indent=2))
        exit(1)
    release_spec = v.normalized(spec)

    # assemble tags and dockerfile used to build required images.
    all_tags = assemble_tag(release_spec, FLAGS.bentoml_version, FLAGS.supported_python_version, partials)
    logger.info(all_tags['devel-python3.6-ubuntu18.04'])


if __name__ == "__main__":
    app.run(main)