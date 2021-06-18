#!/usr/bin/env python3
import errno
import itertools
import logging
import multiprocessing
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy

import json
import docker
import yaml
from absl import flags, app
from cerberus import Validator

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

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
    "hub_repository", None, "Docker Hub repository, only used with --upload_to_hub"
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
    "stop_on_failure",
    False,
    "Stop processing tags if any one build fails. If False or not specified, failures are reported but do not affect "
    "the other images.",
)
flags.DEFINE_boolean(
    "stop_at_generate_dockerfile",
    True,
    "Stop at generating dockerfile",
    short_name="sd",
)

# directory and files
flags.DEFINE_string(
    "dockerfile_dir",
    "./generated",
    "path to generated Dockerfile. Existing files will be deleted with new Dockerfiles",
)
flags.DEFINE_string("partials_dir", "./partials", "Partials directory")
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file")

flags.DEFINE_multi_string(
    "supported_python_version",
    ["3.6", "3.7", "3.8"],
    "Supported Python version",
    short_name='p',
)
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
        add_to_tags:
          type: string
        write_to_dockerfile:
          type: string
        dockerfile_subdirectory:
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


def aggregate_dist_combinations(release_spec, used_dist):
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
    formatter.update({d['set_name']: d['add_to_tags'] for d in dists})
    name = string.format(**formatter)

    if "devel" in name:
        _, dev, rest = name.partition("devel")
        return "-".join([dev, f"python{python_version}", rest[1:]])
    return "-".join([bento_version, f"python{python_version}", name])


def get_dist_spec(dist_specs, tag_spec):
    """
    Extract list of dist spec and args for that dist string:
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


def build_release_args(dists, python_ver):
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
    args["PYTHON_VERSION"] = python_ver
    return args


def get_key_from_dist_spec(dists, key):
    """Get flattened list of certain key in list of dist."""
    return list(itertools.chain(*[d[key] for d in dists if key in d]))


def get_first_key_value(dists, key):
    for d in dists:
        if key in d and d[key] is not None:
            return d[key]
    return None


def generate_tag_metadata(release_spec, bento_ver, supported_py_ver, all_partials):
    """
    Generate all tag metadata based on given spec.

    Args:
        release_spec: Nested dict containing tag spec
        bento_ver: Bentoml release version
        supported_py_ver: Supported python release version
        all_partials: Dict containing every partials with its content.

    Returns:
        Dict of tags and content of Dockerfile.
    """
    tag_metadata = defaultdict(list)
    for package, release_type in release_spec["releases"].items():
        # package: model-server/yatai-service -> release: devel, versioned
        for image_type, target_dists in release_type.items():
            for dist_spec in target_dists["spec"]:
                target_dist = get_dist_spec(release_spec["dist"], dist_spec)
                target_releases = aggregate_dist_combinations(release_spec, target_dist)

                for dist in target_releases:
                    for py_ver in supported_py_ver:
                        tag_args = build_release_args(dist, py_ver)
                        tag_name = build_release_tags(
                            dist_spec, dist, bento_ver, py_ver
                        )
                        used_partials = get_key_from_dist_spec(dist, 'partials')
                        write_to_file = get_first_key_value(dist, 'write_to_dockerfile')
                        dockerfile_contents = merge_partials(
                            release_spec["header"], used_partials, all_partials
                        )

                        tag_metadata[tag_name].append(
                            {
                                'package': package,
                                'release': image_type,
                                'dist_spec': dist_spec,
                                'docker_args': tag_args,
                                'partials': used_partials,
                                'write_to_file': write_to_file,
                                'dockerfile_contents': dockerfile_contents,
                            }
                        )

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
            _simple = full_path[len(partial_path) + 1 : -len(".partial.dockerfile")]
            with open(full_path, "r") as f:
                contents = f.read()
            partials[_simple] = contents
    return partials


def merge_partials(headers, used_partials, all_partials):
    """Merge required partials with headers to generate Dockerfile"""
    used_partials = list(used_partials)
    return "\n\n".join([headers] + [all_partials[u] for u in used_partials])


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def upload_in_background(hub_repository, dock, image, tag):
    """Upload a docker image (to be used by multiprocessing)."""
    image.tag(hub_repository, tag=tag)
    logger.info(dock.images.push(hub_repository, tag=tag))


def main(argv):
    if len(argv) > 1:
        app.UsageError("Too many command line args.")

    # parse manifest.yml
    schema = yaml.safe_load(SPEC_SCHEMA)
    with open(FLAGS.manifest_file, "r") as manifest_file:
        spec = yaml.safe_load(manifest_file)
        manifest_file.close()

    # validate manifest configs and parse partial contents and spec
    partials = get_partials_content(FLAGS.partials_dir)
    v = DockerTagValidator(schema, partials=partials)
    if not v.validate(spec):
        logger.error(f"{FLAGS.manifest_file} is invalid.")
        logger.error(yaml.dump(v.errors, indent=2))
        exit(1)
    release_spec = v.normalized(spec)

    # assemble tags and dockerfile used to build required images.
    all_tags = generate_tag_metadata(
        release_spec, FLAGS.bentoml_version, FLAGS.supported_python_version, partials
    )
    if FLAGS.dump_metadata:
        with open("metadata.json", "w+") as of:
            of.write(json.dumps(all_tags, indent=4))
        of.close()

    # Empty Dockerfile directory when building new Dockerfile
    shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
    mkdir_p(FLAGS.dockerfile_dir)

    # Setup Docker credentials
    docker_client = docker.from_env()
    if FLAGS.upload_to_hub:
        if not FLAGS.hub_username:
            logger.error("> ERROR: please set --hub_username when uploading images.")
            exit(1)
        if not FLAGS.hub_password:
            logger.error("> ERROR: please set --hub_password when uploading images.")
            exit(1)
        if not FLAGS.hub_repository:
            logger.error("> ERROR: please set --hub_repository when uploading images.")
            exit(1)
        docker_client.login(username=FLAGS.hub_username, password=FLAGS.hub_password)

    # each tag has a release_tag and tag_spec containing args, dockerfile content, etc. use --dump_metadata for more
    # information.
    failed_tags, succeeded_tags = [], []
    for release_tag, tag_specs in all_tags.items():
        for spec in tag_specs:
            # Generate required dockerfile for each distro.
            if spec['write_to_file'] is not None:
                path = os.path.join(
                    FLAGS.dockerfile_dir,
                    spec['package'],
                    spec['write_to_file'] + ".dockerfile",
                )
                if os.path.exists(path):
                    continue
                logger.info(f"> Writing to {path} ...")
                if not FLAGS.dry_run:
                    mkdir_p(os.path.dirname(path))
                    with open(path, "w") as of:
                        of.write(spec["dockerfile_contents"])

                if FLAGS.stop_at_generate_dockerfile:
                    logger.info(
                        "> Stopping at generating dockerfile since --stop_at_generate_dockerfile=True"
                    )
                    return

            # Generate temp Dockerfile for docker-py to build, since it needs
            # a file path relative to build context
            temp_dockerfile = os.path.join(
                FLAGS.dockerfile_dir, release_tag + ".tmp.Dockerfile"
            )
            if not FLAGS.dry_run:
                with open(temp_dockerfile, "w") as of:
                    of.write(spec["dockerfile_contents"])
            logger.debug(f">> TMP: writing {temp_dockerfile} ...")

            local_repo_tag = f"{spec['package']}:{release_tag}"
            logger.info(f">> Building {temp_dockerfile} with args:")
            for arg, value in spec['docker_args']:
                logger.info(f">>> {arg}={value}")

            # TODO: test with no_cache.
            # This part is copy-pasta from tensorflow/tensorflow
            tag_failed = False
            image = None
            logs = []
            if not FLAGS.dry_run:
                try:
                    resp = docker_client.api.build(
                        timeout=3600,
                        path='.',
                        nocache=False,
                        dockerfile=temp_dockerfile,
                        buildargs=spec['docker_args'],
                        tag=local_repo_tag,
                    )
                    last_event, image_id = None, None
                    while True:
                        try:
                            output = next(resp).decode('utf-8')
                            json_output = json.loads(output.strip('\r\n'))
                            if 'stream' in json_output:
                                logger.info(json_output['stream'], end='')
                                match = re.search(
                                    r'(^Successfully built |sha256:)([0-9a-f]+)$',
                                    json_output['stream'],
                                )
                                if match:
                                    image_id = match.group(2)
                                last_event = json_output['stream']
                                # collect all log lines into the logs object
                                logs.append(json_output)
                        except StopIteration:
                            logger.info('Docker image build complete.')
                            break
                        except ValueError:
                            logger.error(
                                'Error parsing from docker image build: {}'.format(
                                    output
                                )
                            )
                    # If Image ID is not set, the image failed to built properly. Raise
                    # an error in this case with the last log line and all logs
                    if image_id:
                        image = docker_client.images.get(image_id)
                    else:
                        raise docker.errors.BuildError(last_event or 'Unknown', logs)
                except docker.errors.BuildError as e:
                    logger.error(
                        f'>> {local_repo_tag} failed to build with message: "{e.msg}"'
                    )
                    logger.info('>> Build logs follow:')
                    log_lines = [l.get('stream', '') for l in e.build_log]
                    logger.info(''.join(log_lines))
                    failed_tags.append(local_repo_tag)
                    tag_failed = True
                    if FLAGS.stop_on_failure:
                        logger.info('>> ABORTING due to --stop_on_failure!')
                        exit(1)

                # Clean temporary dockerfiles if they were created earlier
                os.remove(temp_dockerfile)

                # Upload new images to DockerHub as long as they built + passed tests
                if FLAGS.upload_to_hub:
                    if tag_failed:
                        continue

                    logger.info(f'>> Uploading to {FLAGS.hub_repository}:{release_tag}')
                    if not FLAGS.dry_run:
                        p = multiprocessing.Process(
                            target=upload_in_background,
                            args=(
                                FLAGS.hub_repository,
                                docker_client,
                                image,
                                release_tag,
                            ),
                        )
                        p.start()

                if not tag_failed:
                    succeeded_tags.append(release_tag)

    if failed_tags:
        logger.error(
            f"> Some tags failed to build or failed testing, check scroll back for errors: {','.join(failed_tags)}"
        )
        exit(1)

    for tag in succeeded_tags:
        logger.info(tag)


if __name__ == "__main__":
    app.run(main)
