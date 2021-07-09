#!/usr/bin/env python3
import itertools
import json
import operator
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, MutableMapping, Optional, Tuple

from absl import app, logging
from dotenv import load_dotenv
from jinja2 import Environment
from requests import Session
from utils import (
    FLAGS,
    ColoredFormatter,
    cached_property,
    flatten,
    get_data,
    get_nested,
    load_manifest_yaml,
    mapfunc,
    maxkeys,
    mkdir_p,
    pprint,
    set_data,
    sprint,
    walk,
)

from docker import DockerClient
from docker.errors import APIError, BuildError, ImageNotFound

README_TEMPLATE: Path = Path('./templates/docs/README.md.j2')

DOCKERFILE_NAME: str = "Dockerfile"
DOCKERFILE_TEMPLATE_SUFFIX: str = ".Dockerfile.j2"
DOCKERFILE_BUILD_HIERARCHY: List[str] = ["base", "runtime", "cudnn", "devel"]
DOCKERFILE_NVIDIA_REGEX: re.Pattern = re.compile(r'(?:nvidia|cuda|cudnn)+')

SUPPORTED_PYTHON_VERSION: List[str] = ["3.6", "3.7", "3.8"]

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
    log.debug("Creating an instance of DockerClient")
    docker_client = DockerClient(
        base_url="unix://var/run/docker.sock", timeout=3600, tls=True
    )


class LogsMixin(object):
    """Utilities for docker build process to use as Mixin"""

    def build_logs(self, resp: Iterator, image_tag: str):
        """
        Stream build logs to stderr.

        Args:
            resp (:obj:`Iterator`):
                blocking generator from docker.api.build
            image_tag (:obj:`str`):
                given model server tags.
                Ex: model-server:0.13.0-python3.8-slim-runtime

        Raises:
            docker.errors.BuildErrors:
                When errors occurs during build process. Usually
                this comes when generated Dockerfile are incorrect.
        """
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
                    output = next(resp).decode('utf-8')
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
    def push_logs(resp: Iterator, image_id: str):
        """
        Stream push logs to stderr

        Args:
            resp (:obj:`Iterator`):
                blocking generator from docker.api.push(stream=True, decode=True)
            image_id (:obj:`str`):
                id of docker.Images. This is the reference of
                our built images.

        Raises:
            docker.errors.APIError:
                Error during push process. This could be timeout, urllib3.exceptions,
                etc.
        """

        try:
            while True:
                try:
                    data = next(resp)
                    status = data.get('status')
                    if 'id' not in data:
                        sprint(status)
                    else:
                        _id = data.get('id')
                        if 'exists' in status:
                            log.warning(f'{_id}: {status}')
                            continue
                        elif 'Pushed' in status:
                            log.debug(f"{status} {_id}")
                        else:
                            progress = data.get('progress')
                            sprint(f"{status} {_id}: {progress}")
                except StopIteration:
                    log.info(
                        f"Successfully pushed {docker_client.images.get(image_id)}."
                    )
                    break
        except APIError as e:
            log.error(f"Errors during `docker push`: {e.response}")


class GenerateMixin(object):
    """Utilities for generation of images metadata to be used as mixin."""

    @staticmethod
    def cudnn(cuda_components: Dict):
        """
        Return cuDNN context

        Args:
            cuda_components (:obj:`Dict[str, Union[Dict[str, ...], str, List[str]]`):
                CUDA dependencies definitions.
        """
        _cudnn = ''
        for k in cuda_components.keys():
            if re.match(r"(cudnn)(\w+)", k):
                _cudnn = k
                break
        cudnn_release_version: str = cuda_components.pop(_cudnn)
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
            input_path (:obj:`pathlib.Path`):
                List of input path
            output_path (:obj:`pathlib.Path`):
                Output path
            metadata (:obj:`Dict[str, Union[str, List[str], Dict[str, ...]]]`):
                templates context
            build_tag (:obj:`Optional[str]`):
                strictly use for FROM args for base image.
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
            additional_args (:obj:`Optional[Dict[str, str]]`):
                optional variables to pass into build context.

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
        """Generate given BentoML release tag."""
        release_type = kwargs['release_type']
        tag_spec = self.specs['tag']

        _core_fmt = kwargs['fmt'] if kwargs.get('fmt') else tag_spec['fmt']
        formatter = {k: kwargs[k] for k in tag_spec.keys() if k != 'fmt'}

        _tag = _core_fmt.format(**formatter)

        if release_type in ['runtime', 'cudnn']:
            formatter['release_type'] = self._default_args['BENTOML_VERSION']
            _tag = "-".join([_core_fmt.format(**formatter), release_type])

        # construct build_tag which is our base_image with $PYTHON_VERSION formats
        formatter.update({"python_version": "$PYTHON_VERSION", "release_type": "base"})
        _build_tag = _core_fmt.format(**formatter)

        return _tag, _build_tag

    def aggregate_dists_releases(self, package: str) -> List[Tuple[str, str]]:
        """Aggregate all combos from manifest.yml"""
        return [
            (rel, d) for rel, dists in self.packages[package].items() for d in dists
        ]

    @lru_cache(maxsize=32)
    def dockerfiles(self, pkg: str, distro: str, pyv: str) -> Dict:
        """
        Generate template context for each distro releases.
        Args:
            pkg: release bentoml packages, either model-server or yatai-service
            distro: linux distro. eg: centos
            pyv: python version
        Returns:
            build context that can be used with self.render
        """

        _cuda_comp: Dict = get_data(self.cuda, self._cuda_version).copy()  # fmt: skip
        # TODO: Better type annotation for nested dict.
        _ctx: Dict = self.releases[distro].copy()

        _ctx['package'] = pkg

        # setup envars
        _ctx['envars'] = self.envars()(_ctx['envars'])

        # set PYTHON_VERSION envars.
        set_data(_ctx, pyv, 'envars', 'PYTHON_VERSION')  # fmt: skip

        # setup cuda deps
        if _ctx["cuda_prefix_url"]:
            major, minor, _ = self._cuda_version.rsplit(".")
            _ctx["cuda"] = {
                "requires": _ctx['cuda_requires'],
                "ml_repo": NVIDIA_ML_REPO_URL.format(_ctx["cuda_prefix_url"]),
                "base_repo": NVIDIA_REPO_URL.format(_ctx["cuda_prefix_url"]),
                "version": {
                    "major": major,
                    "minor": minor,
                    "shortened": f"{major}.{minor}",
                },
                "components": _cuda_comp,
                "cudnn": self.cudnn(_cuda_comp),
            }
            _ctx.pop('cuda_prefix_url')
            _ctx.pop('cuda_requires')
        else:
            log.debug(f"CUDA is disabled for {distro}. Skipping...")
            for key in _ctx.keys():
                if DOCKERFILE_NVIDIA_REGEX.match(key):
                    _ctx.pop(key)

        return _ctx

    @lru_cache(maxsize=4)
    def metadata(self, output_dir: str, pyv: str):
        """Setup templates context"""
        _paths: Dict[str, str] = {}
        _tags: Dict = defaultdict()

        if os.geteuid() == 0:
            log.warning(
                "Detected running as ROOT. Make sure not to "
                "use --generate dockerfiles while using "
                "manager_images. If you need to regenerate "
                "Dockerfiles use manager_dockerfiles instead."
            )

        for pkg in self.packages.keys():
            log.info(f"Generate context for {pkg} python{pyv}")
            for (release, distro_version) in self.aggregate_dists_releases(pkg):
                log.debug(f"Tag context for {release}:{distro_version}")

                # setup our build context
                _tag_ctx = self.dockerfiles(pkg, distro_version, pyv)
                template_dir = Path(_tag_ctx['templates_dir'])

                # generate our image tag.
                _release_tag, _build_tag = self.image_tags(
                    python_version=pyv,
                    release_type=release,
                    suffixes=_tag_ctx['add_to_tags'],
                )

                # setup image tags.
                tag_keys = f"{_tag_ctx['package']}:{_release_tag}"

                # output path.
                base_output_path = Path(output_dir, pkg, f"{distro_version}")
                if release != 'base':
                    output_path = Path(base_output_path, release)
                else:
                    output_path = base_output_path

                # input path.
                input_paths = [str(f) for f in walk(template_dir) if release in f.name]
                if 'rhel' in template_dir.name:
                    if 'cudnn' in release:
                        input_paths = [
                            str(f)
                            for f in template_dir.iterdir()
                            if DOCKERFILE_NVIDIA_REGEX.match(f.name)
                        ]
                        _tag_ctx.update({"docker_build_path": str(output_path)})

                full_output_path = str(Path(output_path, DOCKERFILE_NAME))

                # # setup directory correspondingly, and add these paths
                set_data(_tags, _tag_ctx.copy(), tag_keys)
                set_data(_tags, input_paths, tag_keys, 'input_paths')
                set_data(_tags, str(output_path), tag_keys, 'output_path')
                if release != 'base':
                    set_data(_tags, _build_tag, tag_keys, 'build_tags')

                # setup generated path with tag releases
                set_data(_paths, full_output_path, tag_keys)

        return _tags, _paths

    def generate_readmes(self, output_dir: str, paths: Dict[str, str]):
        """
        Generate README output. We will also handle README context in memory
        Args:
            output_dir (:obj:`str`):
                target directory
            paths (:obj:`Dict[str, str]`):
                Dictionary of release tags and target dockerfile.
        """

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
        if not Path(output_dir, "README.md").exists():
            self.render(
                input_path=README_TEMPLATE,
                output_path=Path(output_dir),
                metadata=_readme_context,
            )

    def generate_dockerfiles(self):
        """
        Generate results targets from given context. This should only
        be used with --generate dockerfiles
        """

        if FLAGS.overwrite:
            # Consider generated directory ephemeral
            log.info(f"Removing dockerfile dir: {FLAGS.dockerfile_dir}\n")
            shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
            mkdir_p(FLAGS.dockerfile_dir)

        for tags, tags_ctx in self._tags.items():
            for inp in tags_ctx['input_paths']:
                build_tag = ''
                if 'build_tags' in tags_ctx.keys():
                    build_tag = tags_ctx['build_tags']

                self.render(
                    input_path=Path(inp),
                    output_path=Path(tags_ctx['output_path']),
                    metadata=tags_ctx,
                    build_tag=build_tag,
                )

        self.generate_readmes(FLAGS.dockerfile_dir, self._paths)


class BuildMixin(object):
    """Utilities for docker build process to be used as mixin."""

    def build_images(self):
        """Build images."""
        for image_tag, dockerfile_path in self.build_tags():
            log.info(f"Building {image_tag} from {dockerfile_path}")
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
                    self._push_context[image_tag] = docker_client.images.get(image_tag)
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
        """Generate build tags."""
        # We will build and push if args is parsed. Establish some variables.
        if FLAGS.releases and FLAGS.releases in DOCKERFILE_BUILD_HIERARCHY:
            build_tags = itertools.chain(
                self.paths['base'].items(), self.paths[FLAGS.releases].items()
            )
        else:
            build_tags = ((k, v) for i in self.paths.values() for k, v in i.items())
        return build_tags


class PushMixin(object):
    """Utilities for push tasks to be used as mixin."""

    def push_tags(self):
        """Generate push tags"""
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

    def push_images(self):
        """Push images to registries"""

        if not self._push_context:
            log.debug("--generate images is not specified. Generate push context...")
            for image_tag, _ in self.build_tags():
                self._push_context[image_tag] = docker_client.images.get(image_tag)

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

            if FLAGS.readmes:
                log.info("--readmes is specified. Exit after pushing readmes.")
                return

            # Then push image to registry.
            for image_tag in self.push_tags():
                image = self._push_context[image_tag]
                reg, tag = image_tag.split(":")
                registry = ''.join(
                    [v for k, v in registry_spec['registry'].items() if reg in k]
                )
                log.info(f"Uploading {image_tag} to {registry}")

                # NOTES: about concurrent pushing
                #   This would change most of our build logics
                #   since DockerClient is essentially a requests.Session,
                #   which doesn't have support for asynchronous requests.
                #   If we want to implement aiohttp then we might want to
                #   run docker from shell commands.
                self.background_upload(image, tag, registry)

                # separate release latest tags for yatai-service
                if all(map(image_tag.__contains__, ['yatai-service', '3.8', 'slim'])):
                    log.info(f"Uploading {image_tag} as latest to {registry}")
                    tag = 'latest'
                    self.background_upload(image, tag, registry)

    def push_readmes(
        self,
        api_url: str,
        repo_url: str,
        readme_path: Path,
        login_payload: Dict[str, str],
    ):
        """
        Push readmes to registries. Currently only supported for hub.docker.com

        Args:
            api_url (:obj:`str`):
                defined under ``manifest.yml``
            repo_url (:obj:`str`):
                defined under ``manifest.yml``
            readme_path (:obj:`pathlib.Path`):
                generated readmes path
            login_payload (:obj:`Dict[str, str]`):
                login credentials for given API
        """

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
            # hub.docker.com specifics
            data = {"full_description": f.read()}
            self.patch(
                repo_url,
                data=json.dumps(data),
                headers=self.headers,
                cookies=logins.cookies,
            )

    def background_upload(self, image, tag, registry):
        """Upload a docker image."""
        image.tag(registry, tag=tag)
        log.debug(f"Pushing {repr(image)} to {registry}...")
        resp = docker_client.images.push(registry, tag=tag, stream=True, decode=True)
        self.push_logs(resp, image.id)

    def auth_registries(self):
        """Setup auth registries for different registries"""

        # https://docs.docker.com/registry/spec/auth/token/
        pprint('Configure Docker')
        try:
            for registry, metadata in self.repository.items():
                if not os.getenv(metadata['pwd']):
                    log.warn(
                        f"If you are intending to use {registry}, make sure to set both "
                        f"{metadata['pwd']} and {metadata['user']} under {os.getcwd()}/.env. Skipping..."
                    )
                    continue
                else:
                    _user = os.getenv(metadata['user'])
                    _pwd = os.getenv(metadata['pwd'])

                    log.info(
                        f"Logging into Docker registry {registry} with credentials..."
                    )
                    docker_client.api.login(
                        username=_user, password=_pwd, registry=registry
                    )
        except APIError:
            log.info(
                "TLS handshake timeout. You can try resetting docker daemon."
                "If you are using systemd you can do `systemctl restart docker`"
            )
            docker_client.from_env(timeout=FLAGS.timeout)


class ManagerClient(Session, LogsMixin, GenerateMixin, BuildMixin, PushMixin):
    """
    BentoService Images Manager. ManagerClient will be run inside a
    docker container where we will generate correct dockerfiles, build
    given images, then push it to given registry defined under manifest.yml

    class attributes:

        - _paths (:obj:`MutableMapping`):
            mapping paths of all docker container id to input Dockerfile

        - _tags (:obj:`MutableMapping`):
            releases image tags metadata. Run

            .. code-block:: shell

                manager_dockerfiles --bentoml_version 0.13.0 --generate dockerfiles --dump_metadata

            to inspect given metadata

        - _push_context (:obj:`Dict`):

            contains mapping for docker.Images object from docker-py with correct releases type

    """

    _paths: MutableMapping = defaultdict()
    _tags: MutableMapping = defaultdict()
    _push_context: Dict = {}

    _template_env: Environment = Environment(
        extensions=["jinja2.ext.do"], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec: Dict, cuda_version: str):
        super(ManagerClient, self).__init__()

        # We will also default bentoml_version to FLAGs.bentoml_version
        self._default_args: Dict = {"BENTOML_VERSION": FLAGS.bentoml_version}

        # namedtuple are slower than just having class objects since we are dealing
        # with a lot of nested properties.
        for key, values in release_spec.items():
            super().__setattr__(f"{key}", values)

        mapfunc(lambda x: list(flatten(x)), self.packages.values())

        # Workaround adding base to multistage enabled distros. This is very janky.
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
        pprint('Setting up metadata')
        for python_version in self.supported_python:
            _tags, _paths = self.metadata(FLAGS.dockerfile_dir, python_version)
            self._tags.update(_tags)
            self._paths.update(_paths)

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

    def generate(self, dry_run: bool = False):
        if os.geteuid() != 0 and not dry_run:
            self.generate_dockerfiles()

    def build(self, dry_run: bool = False):
        if not dry_run:
            self.build_images()

    def push(self, dry_run: bool = False):
        # https://docs.docker.com/docker-hub/download-rate-limit/
        if not dry_run:
            self.auth_registries()
            self.push_images()


def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    # validate releases args.
    if FLAGS.releases and FLAGS.releases not in DOCKERFILE_BUILD_HIERARCHY:
        log.critical(
            f"Invalid --releases arguments. Allowed: {DOCKERFILE_BUILD_HIERARCHY}"
        )

    try:
        # Parse specs from manifest.yml and validate it.
        pprint(f'Validating {FLAGS.manifest_file}')
        release_spec = load_manifest_yaml(FLAGS.manifest_file)
        if FLAGS.validate:
            return

        # Setup manager client.
        manager = ManagerClient(release_spec, FLAGS.cuda_version)

        # generate templates to correct directory.
        if 'dockerfiles' in FLAGS.generate:
            manager.generate(FLAGS.dry_run)

        # Build images.
        if 'images' in FLAGS.generate:
            manager.build(FLAGS.dry_run)

        # Push images when finish building.
        if FLAGS.push:
            manager.push(dry_run=FLAGS.dry_run)
    except KeyboardInterrupt:
        log.info("Stopping now...")
        exit(1)


if __name__ == "__main__":
    app.run(main)
