#!/usr/bin/env python3
import functools
import itertools
import json
import multiprocessing
import operator
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, MutableMapping, Optional, Union

import absl.logging as logging
import urllib3
from absl import app
from dotenv import load_dotenv
from jinja2 import Environment
from requests import Session
from retry import retry

from docker import DockerClient
from docker.errors import APIError, BuildError, ImageNotFound
from utils import *

# vars
README_TEMPLATE: Path = Path('./templates/docs/README.md.j2')

DOCKERFILE_NAME: str = "Dockerfile"
DOCKERFILE_TEMPLATE_SUFFIX: str = ".Dockerfile.j2"
DOCKERFILE_BUILD_HIERARCHY: List[str] = ["base", "runtime", "cudnn", "devel"]
DOCKERFILE_NVIDIA_REGEX: re.Pattern = re.compile(r'(?:nvidia|cuda|cudnn)+')

SUPPORTED_PYTHON_VERSION: List[str] = ["3.7", "3.8"]

NVIDIA_REPO_URL: str = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL: str = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"
)

HTTP_RETRY_ATTEMPTS: int = 2
HTTP_RETRY_WAIT_SECS: int = 20

# setup some default
logging.get_absl_handler().setFormatter(ColoredFormatter())
log = logging.get_absl_logger()

if not os.path.exists('./.env'):
    log.warning(f"Make sure to create .env file at {os.getcwd()}")
else:
    load_dotenv()

if os.geteuid() == 0:
    # We only use docker_client when running as root.
    docker_client = DockerClient("unix://var/run/docker.sock")


class LogsMixin(object):
    """Logs Mixin for docker build process"""

    def build_logs(self, resp: Iterator, image_tag: str):
        last_event: str = ""
        image_id: str = ""
        output: str = ""
        logs: List = []
        built_regex = re.compile(r'(^Successfully built |sha256:)([0-9a-f]+)$')
        try:
            while True:
                try:
                    # output logs to stdout
                    # https://docker-py.readthedocs.io/en/stable/user_guides/multiplex.html
                    output: str = next(resp).decode('utf-8')
                    json_output: Dict = json.loads(output.strip('\r\n'))
                    # output to stderr when running in docker
                    if 'stream' in json_output:
                        sprint(json_output['stream'])
                        matched = built_regex.search(json_output['stream'])
                        if matched:
                            image_id = matched.group(2)
                        last_event = json_output['stream']
                        logs.append(json_output)
                except StopIteration:
                    log.info(f"Successfully built {image_tag}.")
                    break
                except ValueError:
                    log.error(f"Errors while building image:\n{output}")
            if image_id:
                self._push_context[image_tag] = docker_client.images.get(image_id)
            else:
                raise BuildError(last_event or 'Unknown', logs)
        except BuildError as e:
            log.error(f"Failed to build {image_tag} :\n{e.msg}")
            for line in e.build_log:
                if 'stream' in line:
                    sprint(line['stream'].strip())
            log.fatal('ABORTING due to failure!')

    @staticmethod
    def push_logs(resp: Iterator):
        # resp will be docker_client.push(repo, stream=True, decode=True)
        logs: Dict[str, Dict] = defaultdict()
        image_id: str = ""
        try:
            while True:
                try:
                    output: Dict = next(resp)
                    if 'id' in output:
                        image_id = output['id']
                        if 'progress' in output:
                            logs[image_id] = {
                                'progress': output['progress'],
                                'status': output['status'],
                            }
                            sprint(f"{image_id}: {logs[image_id]['progress']}")
                    if 'status' in output and output['status'] == "Pushed":
                        raise StopIteration
                except StopIteration:
                    log.info(
                        f"Successfully pushed {docker_client.images.get(image_id)}"
                    )
                    break
        except APIError as e:
            log.error(f"Errors during `docker push`: {e.response}")
        except urllib3.exceptions.ReadTimeoutError:
            log.warning(f'{image_id} failed to build due to `urllib3.ReadTimeoutError.')


class GenerateMixin(object):
    """Mixin for all generation-related."""

    @staticmethod
    def cudnn_context(cuda_components: Dict):
        """
        Return cuDNN context
        Args:
            cuda_components: dict of CUDA dependencies definitions.
        """
        _cudnn = ''
        for k in cuda_components.keys():
            if re.match(r"(cudnn)(\w+)", k):
                _cudnn = k
                break
        cudnn_release_version: str = cuda_components[_cudnn]
        cudnn_major_version: str = cudnn_release_version.split('.')[0]
        return {
            "version": cudnn_release_version,
            "major_version": cudnn_major_version,
        }

    def render(
        self,
        input_path: Path,
        output_path: Path,
        metadata: Dict,
        build_tag: Optional[str] = None,
    ):
        """
        Render .j2 templates to output path
        Args:
            input_path: List of input path
            output_path: Output path
            metadata: templates context
            build_tag: strictly use for FROM args for base image.
        """
        if DOCKERFILE_TEMPLATE_SUFFIX not in input_path.name:
            output_name = input_path.stem
        else:
            output_name = DOCKERFILE_NAME
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        with input_path.open('r') as inf:
            template = self._template_env.from_string(inf.read())
        inf.close()

        with output_path.joinpath(output_name).open('w') as ouf:
            ouf.write(template.render(metadata=metadata, build_tag=build_tag))
        ouf.close()

    def envars(self, additional_args: Optional[Dict[str, str]] = None):
        """
        Create a dictionary of args that will be parsed during runtime
        Args:
            additional_args: optional variables to pass into build context.
        Returns:
            Dict of args
        """
        _args: Dict = {} if not additional_args else additional_args
        _args.update(self._default_args)

        def update_args_dict(updater):
            # Args will have format ARG=foobar
            if isinstance(updater, str):
                _arg, _, _value = updater.partition("=")
                _args[_arg] = _value
                return
            elif isinstance(updater, list):
                for v in updater:
                    update_args_dict(v)
            elif isinstance(updater, dict):
                for _arg, _value in updater.items():
                    _args[_arg] = _value
            else:
                log.error(f"cannot add to args dict with unknown type {type(updater)}")
            return _args

        return update_args_dict

    def image_tags(self, **kwargs: str):
        """
        Generate given BentoML release tag.
        """
        release_type = kwargs['release_type']
        tag_spec = self.specs['tag']

        _core_fmt = kwargs['fmt'] if kwargs.get('fmt') else tag_spec['fmt']
        formatter = {k: kwargs[k] for k in tag_spec.keys() if k != 'fmt'}

        _tag = _core_fmt.format(**formatter)
        if release_type in ['runtime', 'cudnn']:
            set_data(formatter, self._default_args['BENTOML_VERSION'], 'release_type')
            _tag = "-".join([_core_fmt.format(**formatter), release_type])

        if release_type == 'base':
            return _tag, ''

        # construct build_tag which is our base_image with $PYTHON_VERSION formats
        formatter.update({"python_version": "$PYTHON_VERSION", "release_type": "base"})
        _build_tag = _core_fmt.format(**formatter)

        return _tag, _build_tag

    @functools.lru_cache(maxsize=10)
    def aggregate_dists_releases(self, package: str) -> Iterator:
        """Aggregate all combos from manifest.yml"""
        return [
            (rel, d) for rel, dists in self.packages[package].items() for d in dists
        ]

    @functools.lru_cache(maxsize=100)
    def dockerfile_context(
        self, package: str, distro_version: str, python_version: str,
    ) -> Dict:
        """
        Generate template context for each distro releases.
        Args:
            package: release bentoml packages, either model-server or yatai-service
            distro_version: linux distro. eg: centos
            python_version: python version
        Returns:
            build context that can be used with self.render
        """
        # TODO: also better type hint
        _context: Dict[str, Union[str, Dict[str, Union[str, Dict[str, str]]]]] = {}
        _cuda_components = get_data(self.cuda, self._cuda_version).copy()
        _distro_context = self.releases[distro_version].copy()

        _context.update(_distro_context)
        _context['package'] = package

        # setup envars
        set_data(
            _context, self.envars()(_distro_context['envars']), 'envars',
        )

        # set PYTHON_VERSION envars.
        set_data(
            _context, python_version, 'envars', 'PYTHON_VERSION',
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
                "components": _cuda_components,
                "cudnn": self.cudnn_context(_cuda_components),
            }
            _context.pop('cuda_prefix_url')
        else:
            log.debug(f"CUDA is disabled for {distro_version}. Skipping...")
            for key in list(_context.keys()):
                if DOCKERFILE_NVIDIA_REGEX.match(key):
                    _context.pop(key)

        return _context

    def readmes(self, target_dir: str, paths: Dict):
        # this will include both yatai-service and models-server path.
        _release_context: MutableMapping = defaultdict(list)

        # check if given distro is supported per package.
        for distro_version in self.releases.keys():
            if 'debian' in distro_version:
                _os_tag = 'slim'
            elif 'amazonlinux' in distro_version:
                _os_tag = 'ami'
            else:
                _os_tag = distro_version

            _release_context[distro_version] = sorted(
                [
                    (release_tags, gen_path)
                    for release_tags, gen_path in paths.items()
                    if _os_tag in release_tags and 'base' not in release_tags
                ],
                key=operator.itemgetter(0),
            )

        _readme_context: Dict = {
            'ephemeral': False,
            'bentoml_package': "",
            'bentoml_release_version': FLAGS.bentoml_version,
            'release_info': _release_context,
        }

        for package in self.packages.keys():
            output_readme = Path('docs', package)
            shutil.rmtree(package, ignore_errors=True)

            set_data(_readme_context, package, 'bentoml_package')
            _readme_context['oss'] = maxkeys(self.packages[package])

            self.render(
                input_path=README_TEMPLATE,
                output_path=output_readme,
                metadata=_readme_context,
            )

        # renders README for generated directory
        _readme_context['ephemeral'] = True
        if not Path(target_dir, "README.md").exists():
            self.render(
                input_path=README_TEMPLATE,
                output_path=Path(target_dir),
                metadata=_readme_context,
            )

    def metadata(self, target_dir: str, python_versions: List[str]):
        _paths: Dict[str, str] = {}
        _tags: Dict = defaultdict()

        if os.geteuid() == 0:
            log.warning(
                "Detected running as ROOT. Make sure "
                "to use manager_dockerfiles to generate Dockerfile"
            )
        pprint('Setting up metadata')
        for package, python_version in itertools.product(
            self.packages.keys(), python_versions
        ):
            log.info(f"Working on {package} for python{python_version}")
            for (release, distro_version) in self.aggregate_dists_releases(package):
                log.debug(f"Tag context: {distro_version}:{release}")
                # setup our build context
                tag_context = self.dockerfile_context(
                    package, distro_version, python_version,
                )
                template_dir = Path(tag_context['templates_dir'])

                # generate our image tag.
                _release_tag, _build_tag = self.image_tags(
                    python_version=python_version,
                    release_type=release,
                    suffixes=tag_context['add_to_tags'],
                )

                # input paths
                input_paths = [f for f in walk(template_dir) if release in f.name]
                if 'rhel' in template_dir.name and release == 'cudnn':
                    input_paths = [
                        f
                        for f in template_dir.iterdir()
                        if DOCKERFILE_NVIDIA_REGEX.match(f.name)
                    ]

                # output paths
                output_path = Path(target_dir, package, f"{distro_version}")
                if release != 'base':
                    output_path = Path(output_path, release)
                if release == 'cudnn':
                    tag_context.update({"docker_build_path": str(output_path)})

                full_output_path = str(Path(output_path, DOCKERFILE_NAME))

                tag_keys = f"{tag_context['package']}:{_release_tag}"

                # # setup directory correspondingly, and add these paths
                set_data(_tags, tag_context, tag_keys)
                set_data(_paths, full_output_path, tag_keys)
                if 'dockerfiles' in FLAGS.generate and os.geteuid() != 0:
                    # Forbid generation by ROOT
                    for input_path in input_paths:
                        self.render(
                            input_path=input_path,
                            output_path=output_path,
                            metadata=tag_context,
                            build_tag=_build_tag,
                        )
                    self.readmes(FLAGS.dockerfile_dir, _paths)
        return _tags, _paths


class BuildMixin(object):
    """Mixin for Build-related."""

    def build_images(self):
        for image_tag, dockerfile_path in self.build_tags():
            log.debug(f"Building {image_tag} from {dockerfile_path}")
            try:
                # this is should be when there are changed in base image.
                if FLAGS.overwrite:
                    docker_client.api.remove_image(image_tag, force=True)
                    # workaround for not having goto supports in Python.
                    raise ImageNotFound(f"Building {image_tag}")
                if docker_client.api.history(image_tag):
                    log.info(
                        f"Image {image_tag} is already built and --overwrite is not specified. Skipping..."
                    )
                    set_data(
                        self._push_context,
                        docker_client.images.get(image_tag),
                        image_tag,
                    )
                    continue
            except ImageNotFound:
                pyenv = get_data(self._tags, image_tag, 'envars', 'PYTHON_VERSION')
                build_args = {"PYTHON_VERSION": pyenv}
                # NOTES: https://warehouse.pypa.io/api-reference/index.html
                resp = docker_client.api.build(
                    timeout=FLAGS.timeout,
                    path=".",
                    nocache=False,
                    buildargs=build_args,
                    dockerfile=dockerfile_path,
                    tag=image_tag,
                )
                self.build_logs(resp, image_tag)

    def build_tags(self):
        # We will build and push if args is parsed. Establish some variables.
        if FLAGS.releases and FLAGS.releases in DOCKERFILE_BUILD_HIERARCHY:
            build_tags = itertools.chain(
                self.paths['base'].items(), self.paths[FLAGS.releases].items()
            )
        else:
            build_tags = ((k, v) for i in self.paths.values() for k, v in i.items())
        return build_tags


class PushMixin(object):
    """Mixin for Push-related."""

    def push_tags(self):
        _paths = deepcopy(self.paths)
        if FLAGS.releases and FLAGS.releases in DOCKERFILE_BUILD_HIERARCHY:
            _release_tag = _paths[FLAGS.releases]
        else:
            # we don't need to publish base tags.
            _paths.pop('base')
            _release_tag = flatten(
                map(lambda x: [_x for _x in x.keys()], _paths.values())
            )
        return _release_tag

    def push_to_registries(self):

        self.auth_registries()

        if not self._push_context:
            log.debug("--generate images is not specified. Generate push context...")
            for image_tag, dockerfile_path in self.build_tags():
                set_data(
                    self._push_context, docker_client.images.get(image_tag), image_tag,
                )

        for registry, registry_spec in self.repository.items():
            # We will want to push README first.
            login_payload: Dict = {
                "username": os.getenv(registry_spec['user']),
                "password": os.getenv(registry_spec['pwd']),
            }
            api_url: str = get_nested(registry_spec, ['urls', 'api'])

            for package, registry_url in registry_spec['registry'].items():
                _, _url = registry_url.split('/', maxsplit=1)
                readme_path = Path('docs', package, 'README.md')
                repo_url: str = f"{get_nested(registry_spec, ['urls', 'repos'])}/{_url}/"
                self.push_readmes(api_url, repo_url, readme_path, login_payload)

            # Then push image to registry.
            for image_tag in self.push_tags():
                image = self._push_context[image_tag]
                reg, tag = image_tag.split(":")
                registry = ''.join((k for k in registry_spec['registry'] if reg in k))
                log.info(f"Uploading {image_tag} to {registry}")
                p = multiprocessing.Process(
                    target=self.upload_in_background, args=(image, tag, registry),
                )
                p.start()

    def push_readmes(
        self,
        api_url: str,
        repo_url: str,
        readme_path: Path,
        login_payload: Dict[str, str],
    ):
        self.headers.update(
            {"Content-Type": "application/json", "Connection": "keep-alive"}
        )
        logins = self.post(
            api_url, data=json.dumps(login_payload), headers=self.headers
        )
        self.headers.update(
            {
                "Authorization": f"JWT {logins.json()['token']}",
                "Access-Control-Allow-Origin": "*",
                "X-CSRFToken": logins.cookies['csrftoken'],
            }
        )
        with readme_path.open('r') as f:
            data = {"full_description": f.read()}
            self.patch(
                repo_url,
                data=json.dumps(data),
                headers=self.headers,
                cookies=logins.cookies,
            )

    @retry(PushRetry, tries=HTTP_RETRY_ATTEMPTS, delay=HTTP_RETRY_WAIT_SECS, logger=log)
    def upload_in_background(self, image, tag, registry):
        """Upload a docker image (to be used by multiprocessing)."""
        image.tag(registry, tag=tag)
        log.debug(f"Pushing {image.tag} to {registry}...")
        try:
            resp = docker_client.images.push(
                registry, tag=tag, stream=True, decode=True
            )
            self.push_logs(resp)
        except APIError:
            raise PushRetry

    @retry(
        LoginRetry, tries=HTTP_RETRY_ATTEMPTS, delay=HTTP_RETRY_WAIT_SECS, logger=log
    )
    def auth_registries(self):

        # Login Docker credentials and setup docker client
        pprint('Configure Docker')
        try:
            docker_client.from_env(timeout=FLAGS.timeout)
        except APIError:
            for registry, metadata in self.repository.items():
                if not os.getenv(metadata['pwd']):
                    log.critical(
                        f"{registry}: Make sure to"
                        f" set both {metadata['pwd']} and {metadata['user']}."
                    )
                _user = os.getenv(metadata['user'])
                _pwd = os.getenv(metadata['pwd'])

                log.info(f"Logging into Docker registry {registry} with credentials...")
                docker_client.api.login(
                    username=_user, password=_pwd, registry=registry
                )
            raise LoginRetry


class ManagerClient(Session, LogsMixin, GenerateMixin, BuildMixin, PushMixin):
    """
    Docker Images Releases Manager.
    """

    _paths: MutableMapping = defaultdict()
    _tags: MutableMapping = defaultdict()
    _push_context: Dict = {}

    _template_env: Environment = Environment(
        extensions=["jinja2.ext.do"], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec: Dict, cuda_version: str, *args, **kwargs):
        super(ManagerClient, self).__init__(*args, **kwargs)

        # We will also default bentoml_version to FLAGs.bentoml_version
        self._default_args: Dict = {"BENTOML_VERSION": FLAGS.bentoml_version}

        # nametuple are slower than just having class objects since we are dealing
        # with a lot of nested properties.
        for key, values in release_spec.items():
            super().__setattr__(f"{key}", values)

        # Workaround adding base to multistage enabled distros.
        # NOTE: Ugh this is jank
        mapfunc(lambda x: list(flatten(x)), self.packages.values())
        for p in self.packages.values():
            p.update({"base": maxkeys(p)})

        # setup cuda version
        self._cuda_version = cuda_version

        # setup python version.
        if len(FLAGS.python_version) > 0:
            log.debug("--python_version defined, ignoring SUPPORTED_PYTHON_VERSION")
            self.supported_python = FLAGS.python_version
            if self.supported_python not in SUPPORTED_PYTHON_VERSION:
                log.critical(
                    f"{self.supported_python} is not supported. Allowed: {SUPPORTED_PYTHON_VERSION}"
                )
        else:
            self.supported_python = SUPPORTED_PYTHON_VERSION

        # generate Dockerfiles and README.
        if FLAGS.overwrite:
            # Consider generated directory ephemeral
            log.info(f"Removing dockerfile dir: {FLAGS.dockerfile_dir}\n")
            shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
            mkdir_p(FLAGS.dockerfile_dir)
        self._tags, self._paths = self.metadata(
            FLAGS.dockerfile_dir, self.supported_python
        )
        if FLAGS.dump_metadata:
            with open("./metadata.json", "w") as ouf:
                ouf.write(json.dumps(self._tags, indent=2))
            ouf.close()

    @cached_property
    def paths(self):
        return {
            h: {k: self._paths[k] for k in self._paths.keys() if h in k}
            for h in DOCKERFILE_BUILD_HIERARCHY
        }

    def build(self, dry_run: bool = False):
        if not dry_run:
            self.build_images()

    def push(self, dry_run: bool = False):
        if not dry_run:
            self.push_to_registries()


def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    # validate releases args.
    if FLAGS.releases and FLAGS.releases not in DOCKERFILE_BUILD_HIERARCHY:
        log.critical(
            f"Invalid --releases arguments. Allowed: {DOCKERFILE_BUILD_HIERARCHY}"
        )

    # Parse specs from manifest.yml and validate it.
    pprint(f'Validating {FLAGS.manifest_file}')
    release_spec = load_manifest_yaml(FLAGS.manifest_file)

    if FLAGS.validate:
        return

    # generate templates to correct directory.
    manager = ManagerClient(release_spec, FLAGS.cuda_version)
    if 'dockerfiles' in FLAGS.generate:
        return

    # Build images.
    if 'images' in FLAGS.generate:
        manager.build(FLAGS.dry_run)

    # Push images when finish building.
    if FLAGS.push_to_hub:
        manager.push(dry_run=FLAGS.dry_run)


if __name__ == "__main__":
    app.run(main)