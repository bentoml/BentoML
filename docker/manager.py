#!/usr/bin/env python3
import itertools
import json
import multiprocessing
import operator
import os
from pathlib import Path
import re
import shutil
from collections import defaultdict
from typing import Optional, Dict, MutableMapping, List

import absl.logging as logging
import docker
import requests
from absl import app
from dotenv import load_dotenv
from jinja2 import Environment

from utils import (
    FLAGS,
    SUPPORTED_GENERATE_TYPE,
    mkdir_p,
    ColoredFormatter,
    maxkeys,
    mapfunc,
    cached_property,
    flatten,
    walk,
    load_manifest_yaml,
    dprint,
    get_data,
    set_data,
)

load_dotenv()
logging.get_absl_handler().setFormatter(ColoredFormatter())
logger = logging.get_absl_logger()

RELEASE_TAG_FORMAT = "{release_type}-python{python_version}-{tag_suffix}"

DOCKERFILE_SUFFIX = ".Dockerfile.j2"
DOCKERFILE_BUILD_HIERARCHY = ["base", "runtime", "cudnn", "devel"]
DOCKERFILE_NVIDIA_REGEX = re.compile(r'(?:nvidia|cuda|cudnn)+')
DOCKERFILE_NAME = DOCKERFILE_SUFFIX.split('.')[1]

SUPPORTED_PYTHON_VERSION = ["3.7", "3.8"]

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"
)


def auth_registries(repository, docker_client):
    repos = {}
    for registry, metadata in repository.items():
        if not os.getenv(metadata['pwd']) and not os.getenv(metadata['user']):
            logger.critical(
                f"{registry}: Make sure to"
                f" set {metadata['pwd']} or {metadata['user']}."
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
        dprint(json.dumps(_line, indent=2))


# noinspection PyUnresolvedReferences
class GenerateAPIMixin(object):
    """Mixin for all generation-related."""

    def render(
        self,
        input_paths: List[Path],
        output_path: Path,
        metadata: Dict,
        build_tag: Optional[str] = None,
    ):
        """
        Render .j2 templates to output path
        Args:
            input_paths: List of input path
            output_path: Output path
            metadata: templates context
            build_tag: strictly use for FROM args for base image.
        """
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

                with output_path.joinpath(output_name).open('w') as ouf:
                    ouf.write(template.render(metadata=metadata, build_tag=build_tag))
                ouf.close()
            inf.close()

    @staticmethod
    def envars(bentoml_version: str):
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
    def image_tags(
        build_ctx: MutableMapping,
        python_version: str,
        release_type: str,
        fmt: Optional[str] = "",
    ):
        """
        Generate given BentoML release tag.
        Args:
            build_ctx: generated templates context
            python_version: supported python version. check SUPPORTED_PYTHON_VERSION
            release_type: type of releases. devel, runtime, cudnn
            fmt: Custom tag format if needed.
        """
        _core_fmt = fmt if fmt else RELEASE_TAG_FORMAT
        formatter = {
            "release_type": release_type,
            "python_version": python_version,
            "tag_suffix": build_ctx["add_to_tags"],
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
    def cudnn(cuda_components: Dict):
        """
        Return cuDNN context
        Args:
            cuda_components: dict of CUDA dependencies definitions.
        """
        cudnn_regex = re.compile(r"(cudnn)(\w+)")
        for k, v in cuda_components.items():
            if cudnn_regex.match(k):
                cudnn_release_version = cuda_components.pop(k)
                cudnn_major_version = cudnn_release_version.split(".")[0]
                return {
                    "version": cudnn_release_version,
                    "major_version": cudnn_major_version,
                }

    def aggregate_dists_releases(self, package_spec: MutableMapping):
        """Aggregate all combos from manifest.yml"""
        for releases, distros in package_spec.items():
            for dist in distros:
                for distro_version, distro_spec in self._releases[dist].items():
                    yield dist, distro_version, distro_spec, releases

    def dockerfile_context(
        self,
        package: str,
        distro: str,
        distro_version: str,
        distro_spec: Dict[str, str],
        python_version: str,
        bentoml_version: str,
    ) -> Dict:
        """
        Generate template context for each distro releases.
        Args:
            distro: linux distro. eg: centos
            distro_version: linux distro version. eg: 7
            distro_spec: linux distro spec from manifest.yml
            python_version: python version
            bentoml_version: bentoml version
            package: release bentoml packages, either model-server or yatai-service
        Returns:
            build context that can be used with self.render
        """
        _context = {}
        _cuda_components = get_data(self._cuda_dependencies, self._cuda_version).copy()

        set_data(_context, package, 'package')
        set_data(_context, get_data(self._releases, distro, distro_version))

        # setup envars
        set_data(
            _context, self.envars(bentoml_version)(distro_spec['envars']), 'envars',
        )

        # check if python_version is supported.
        set_data(
            _context, python_version, "envars", "PYTHON_VERSION",
        )

        # setup cuda deps
        if _context["cuda_prefix_url"]:
            major, minor, _ = self._cuda_version.rsplit(".")
            _context["cuda"] = {
                "ml_repo": NVIDIA_ML_REPO_URL.format(_context["cuda_prefix_url"]),
                "base_repo": NVIDIA_REPO_URL.format(_context["cuda_prefix_url"]),
                "version": {
                    "major": major,
                    "minor": minor,
                    "shortened": f"{major}.{minor}",
                },
                "cudnn": self.cudnn(_cuda_components),
                "components": _cuda_components,
            }
            _context.pop('cuda_prefix_url')
        else:
            logger.debug(f"CUDA is disabled for {distro}. Skipping...")
            for key in list(_context.keys()):
                if DOCKERFILE_NVIDIA_REGEX.match(key):
                    _context.pop(key)

        return _context

    def release_info_context(
        self, package: str,
    ):
        release_info: MutableMapping = defaultdict(list)

        assert (
            self._paths
        ), "Make sure to generate build metadata before parsing README context."

        # check if given distro is supported per package.
        dists = maxkeys(self._packages[package])
        for distro in dists:
            for distro_version in self._releases[distro].keys():
                combos = f"{distro}{distro_version}"
                if distro == 'debian':
                    _os_tag = combos = 'slim'
                elif distro == 'amazonlinux':
                    _os_tag = 'ami'
                else:
                    _os_tag = distro

                release_info[combos] = sorted(
                    [
                        (release_tags.split(":")[1], gen_path)
                        for release_tags, gen_path in self._paths.items()
                        if _os_tag in release_tags
                        and package in release_tags
                        and 'base' not in release_tags
                        and str(distro_version) in gen_path
                    ],
                    key=operator.itemgetter(0),
                )
        return release_info

    def metadata(self, target_dir: str, python_versions: List[str]):
        for combinations in itertools.product(self._packages.items(), python_versions):
            (package, package_spec), python_version = combinations
            logger.info(f"Working on {package} for python{python_version}")
            for (
                distro,
                distro_version,
                distro_spec,
                release,
            ) in self.aggregate_dists_releases(package_spec):
                # setup our build context
                tag_context = self.dockerfile_context(
                    package,
                    distro,
                    distro_version,
                    distro_spec,
                    python_version,
                    FLAGS.bentoml_version,
                )
                # generate our image tag.
                _release_tag, build_tag = self.image_tags(
                    tag_context, python_version, release,
                )

                template_dir = Path(tag_context['templates_dir'])

                # input paths
                input_paths = [f for f in walk(template_dir) if release in f.name]
                if 'rhel' in template_dir.name and release == 'cudnn':
                    input_paths = [
                        f
                        for f in template_dir.iterdir()
                        if DOCKERFILE_NVIDIA_REGEX.match(f.name)
                    ]

                # output paths
                output_path = Path(target_dir, package, f"{distro}{distro_version}")
                if release != 'base':
                    output_path = Path(output_path, release)
                full_output_path = str(Path(output_path, DOCKERFILE_NAME))

                if release == 'cudnn':
                    tag_context.update({"docker_build_path": str(output_path)})

                tag_keys = f"{tag_context['package']}:{_release_tag}"

                # # setup directory correspondingly, and add these paths
                set_data(self._tags, tag_context, tag_keys)
                set_data(self._paths, full_output_path, tag_keys)

                # generate our Dockerfile from templates.
                if 'dockerfiles' in FLAGS.generate:
                    self.render(
                        input_paths=input_paths,
                        output_path=output_path,
                        metadata=tag_context,
                        build_tag=build_tag,
                    )

    def readmes(self, target_dir: str):
        input_readmes: List[Path] = [Path('./templates/docs/README.md.j2')]
        readme_context: Dict = {
            'ephemeral': False,
            'bentoml_img': 'https://github.com/bentoml/BentoML/blob/master/docs/source/_static/img/bentoml.png',
            'bentoml_package': "",
            'bentoml_release_version': FLAGS.bentoml_version,
        }
        for package in self._packages.keys():
            _release_info = self.release_info_context(package)

            readme_context['release_info'] = _release_info

            set_data(readme_context, package, 'bentoml_package')
            output_readme = Path('docs', package)
            self.render(
                input_paths=input_readmes,
                output_path=output_readme,
                metadata=readme_context,
            )

        # renders README for generated directory
        readme_context['ephemeral'] = True
        self.render(
            input_paths=input_readmes,
            output_path=Path(target_dir),
            metadata=readme_context,
        )


class BuildAPIMixin(object):
    pass


# noinspection PyUnresolvedReferences
class Manager(requests.Session, GenerateAPIMixin, BuildAPIMixin):
    _paths: Dict = {}
    _tags: MutableMapping = defaultdict()

    _template_env: Environment = Environment(
        extensions=["jinja2.ext.do"], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec: Dict, cuda_version: str):
        # set private class attributes to be manifest.yml keys
        super(Manager, self).__init__()
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
            raise RuntimeError(
                f"{cuda_version} is not defined in {FLAGS.manifest_file}"
            )

    def __call__(self, dry_run=False):
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
            # generate Dockerfiles and README.
            if 'dockerfiles' in FLAGS.generate:
                # Consider generated directory ephemeral
                logger.info(f"Removing dockerfile dir: {FLAGS.dockerfile_dir}\n")
                shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
                mkdir_p(FLAGS.dockerfile_dir)
            self.metadata(
                target_dir=FLAGS.dockerfile_dir, python_versions=supported_python
            )
            self.readmes(target_dir=FLAGS.dockerfile_dir)

            if FLAGS.dump_metadata:
                logger.info(
                    "--dump_metadata is specified. Stop after dumping metadata..."
                )
                with open("./metadata.json", "w") as ouf:
                    ouf.write(json.dumps(self._tags, indent=2))
                ouf.close()

        return self

    def __repr__(self):
        return json.dumps(self.paths, indent=2)

    @property
    def paths(self):
        return {
            h: {k: self._paths[k] for k in self._paths.keys() if h in k}
            for h in DOCKERFILE_BUILD_HIERARCHY
        }

    @cached_property
    def tags(self):
        return self._tags


def main(argv):
    push_tags, logs = {}, []

    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    if FLAGS.releases and FLAGS.releases not in SUPPORTED_GENERATE_TYPE:
        logger.critical(
            f"Invalid --releases arguments. Allowed: {SUPPORTED_GENERATE_TYPE}"
        )

    # Parse specs from manifest.yml and validate it.
    logger.info(
        f"{'-' * 59}\n\t{' ' * 5}| Validating {FLAGS.manifest_file}\n\t{' ' * 5}{'-' * 59}"
    )
    release_spec = load_manifest_yaml(FLAGS.manifest_file)

    # generate templates to correct directory.
    logger.info(
        f"{'-' * 59}\n\t{' ' * 5}| Generate from templates\n\t{' ' * 5}{'-' * 59}"
    )
    manager = Manager(release_spec, FLAGS.cuda_version)(dry_run=FLAGS.dry_run)
    tag_metadata, path_metadata = manager.tags, manager.paths

    if 'dockerfiles' in FLAGS.generate:
        # This to stop the process at generating Dockerfile.
        return

    # Login Docker credentials and setup docker client
    logger.info(f"{'-' * 59}\n\t{' ' * 5}| Configure Docker\n\t{' ' * 5}{'-' * 59}")
    docker_client = docker.from_env()

    if len(os.listdir(FLAGS.dockerfile_dir)) == 0:
        logger.critical(
            "Please use manager_dockerfiles --generate dockerfiles"
            " in order to generate Dockerfiles with correct permission."
        )
    if os.geteuid() != 0:
        logger.warning(
            "Make sure to use manager_images to "
            "connect to docker.sock as ROOT. Currently running without sudo privilege."
        )

    # We will build and push if args is parsed. Establish some variables.
    logger.info(f"{'-' * 59}\n\t{' ' * 5}| Building images\n\t{' ' * 5}{'-' * 59}")
    if FLAGS.releases and FLAGS.releases in DOCKERFILE_BUILD_HIERARCHY:
        _build_tags = itertools.chain(
            path_metadata['base'].items(), path_metadata[FLAGS.releases].items()
        )
    else:
        _build_tags = ((k, v) for i in path_metadata.values() for k, v in i.items())

    if 'images' in FLAGS.generate:
        for image_tag, dockerfile_path in _build_tags:
            logger.info(f"Building {image_tag} from {dockerfile_path}")
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
                    build_args = {
                        "PYTHON_VERSION": get_data(
                            tag_metadata, image_tag, 'envars', 'PYTHON_VERSION'
                        )
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
                    output = ""
                    while True:
                        try:
                            # output logs to stdout
                            # https://docker-py.readthedocs.io/en/stable/user_guides/multiplex.html
                            output = next(resp).decode('utf-8')
                            json_output = json.loads(output.strip('\r\n'))
                            if 'stream' in json_output:
                                # output docker build process
                                dprint(json_output['stream'])
                                built = re.compile(
                                    r'(^Successfully built |sha256:)([0-9a-f]+)$'
                                )
                                matched = built.search(json_output['stream'])
                                if matched:
                                    image_id = matched.group(2)
                                last_event = json_output['stream']
                                logs.append(json_output)
                        except StopIteration:
                            logger.info(f"Successfully built {image_tag}.")
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
                            dprint(line['stream'].strip())
                    logger.fatal('ABORTING due to failure!')
        return

    # Push process whether to push base image or not
    # should be able to specified which process to push.
    if FLAGS.push_to_hub:
        for image_tag, dockerfile_path in _build_tags:
            push_tags[image_tag] = docker_client.images.get(image_tag)
        logger.info(f"{'-' * 59}\n\t{' ' * 5}| Pushing images\n\t{' ' * 5}{'-' * 59}")
        repos = auth_registries(release_spec["repository"], docker_client)

        assert push_tags, logger.error("Push tags is empty!!")
        if FLAGS.releases and FLAGS.releases in DOCKERFILE_BUILD_HIERARCHY:
            _release_tag = path_metadata[FLAGS.releases]
        else:
            # we don't need to publish base tags.
            path_metadata.pop('base')
            _release_tag = flatten(
                map(lambda x: [_x for _x in x.keys()], path_metadata.values())
            )
        for repo, registries in repos.items():
            for image_tag in _release_tag:
                image = push_tags[image_tag]
                _registry, tag = image_tag.split(":")
                registry = ''.join((k for k in registries if _registry in k))
                logger.info(f"Uploading {image_tag} to {repo}")
                if not FLAGS.dry_run:
                    p = multiprocessing.Process(
                        target=upload_in_background,
                        args=(docker_client, image, registry, tag),
                    )
                    p.start()


# update README hack
# https://github.com/docker/hub-feedback/issues/2006
# TOKEN=$(curl -s -H "Content-Type: application/json" -X POST -d \
#         '{"username": "'${UNAME}'", "password": "'${UPASS}'"}' https://hub.docker.com/v2/users/logins/ | jq -r .token)
#
# curl -s -vvv -X PATCH -H "Content-Type: application/json" -H "Authorization: JWT ${TOKEN}" \
#       -d '{"full_description": "ed"}' https://hub.docker.com/v2/repositories/bentoml/model-server/

if __name__ == "__main__":
    app.run(main)
