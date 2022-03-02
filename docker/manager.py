#!/usr/bin/env python3
import os
import re
import sys
import json
import shutil
import typing as t
import logging
import operator
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import attrs
from absl import app
from absl import logging as absl_logging
from utils import walk
from utils import FLAGS
from utils import pprint
from utils import sprint
from utils import flatten
from utils import mapfunc
from utils import maxkeys
from utils import get_data
from utils import set_data
from utils import shellcmd
from utils import get_nested
from utils import graceful_exit
from utils import cached_property
from utils import DOCKERFILE_NAME
from utils import NVIDIA_REPO_URL
from utils import README_TEMPLATE
from utils import ColoredFormatter
from utils import load_manifest_yaml
from utils import NVIDIA_ML_REPO_URL
from utils import HTTP_RETRY_ATTEMPTS
from utils import HTTP_RETRY_WAIT_SECS
from utils import DOCKERFILE_NVIDIA_REGEX
from utils import SUPPORTED_PYTHON_VERSION
from utils import DOCKERFILE_BUILD_HIERARCHY
from utils import DOCKERFILE_TEMPLATE_SUFFIX
from utils import SUPPORTED_ARCHITECTURE_TYPE
from dotenv import load_dotenv
from jinja2 import Environment
from requests import Session

from docker import DockerClient
from docker.errors import APIError
from docker.errors import BuildError
from docker.errors import ImageNotFound

if TYPE_CHECKING:
    from docker.models.images import Image  # pylint: disable=unused-import

    DictStrAny = t.Dict[str, t.Any]

# setup some default
handler = t.cast(logging.Handler, absl_logging.get_absl_handler())
handler.setFormatter(ColoredFormatter())
logger = logging.getLogger("absl")

if not os.path.exists("./.env"):
    logger.warning(f"Make sure to create .env file at {os.getcwd()}")
else:
    load_dotenv()

if os.geteuid() == 0:
    # We only use docker_client when running as root.
    logger.debug("Creating an instance of DockerClient")
    docker_client = DockerClient(
        base_url="unix://var/run/docker.sock", timeout=3600, tls=True
    )


@attrs.define
class Manifest:
    ...


def cudnn(cuda_components: t.Dict[str, t.Any]) -> t.Dict[str, str]:
    """
    Return cuDNN context

    Args:
        cuda_components (:obj:`t.Dict[str, Union[t.Dict[str, ...], str, t.List[str]]`):
            CUDA dependencies definitions.
    """
    _cudnn = ""
    for k in cuda_components.keys():
        if re.match(r"(cudnn)(\w+)", k):
            _cudnn = k
            break
    cudnn_release_version: str = cuda_components.pop(_cudnn)
    cudnn_major_version: str = cudnn_release_version.split(".")[0]
    return {
        "version": cudnn_release_version,
        "major_version": cudnn_major_version,
    }


def render(
    self,
    input_path: Path,
    output_path: Path,
    metadata: t.Dict,
    build_tag: t.Optional[str] = None,
) -> None:
    """
    Render .j2 templates to output path

    Args:
        input_path (:obj:`pathlib.Path`):
            t.List of input path
        output_path (:obj:`pathlib.Path`):
            Output path
        metadata (:obj:`t.Dict[str, Union[str, t.List[str], t.Dict[str, ...]]]`):
            templates context
        build_tag (:obj:`t.Optional[str]`):
            strictly use for FROM args for base image.
    """
    if DOCKERFILE_TEMPLATE_SUFFIX not in input_path.name:
        output_name = input_path.stem
    else:
        output_name = DOCKERFILE_NAME
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as inf:
        template = self._template_env.from_string(inf.read())
    inf.close()

    with output_path.joinpath(output_name).open("w") as ouf:
        ouf.write(template.render(metadata=metadata, build_tag=build_tag))
    ouf.close()


def envars(
    self, additional_args: t.Optional[t.Dict[str, str]] = None
) -> t.Callable[[t.Union[str, dict, list]], dict]:
    """
    Create a dictionary of args that will be parsed during runtime

    Args:
        additional_args (:obj:`t.Optional[t.Dict[str, str]]`):
            optional variables to pass into build context.

    Returns:
        t.Dict of args
    """
    _args: dict = {} if not additional_args else additional_args
    _args.update(self._default_args)

    def update_args_dict(updater: t.Union[str, dict, list]) -> dict:
        # Args will have format ARG=foobar
        if isinstance(updater, str):
            _arg, _, _value = updater.partition("=")
            _args[_arg] = _value
        elif isinstance(updater, list):
            for v in updater:
                update_args_dict(v)
        elif isinstance(updater, dict):
            for _arg, _value in updater.items():
                _args[_arg] = _value
        else:
            logger.error(f"cannot add to args dict with unknown type {type(updater)}")
        return _args

    return update_args_dict


def image_tags(self, **kwargs: str) -> t.Tuple[str, str]:
    """Generate given BentoML release tag."""
    release_type = kwargs["release_type"]
    tag_spec = self.specs["tag"]

    _core_fmt = kwargs["fmt"] if kwargs.get("fmt") else tag_spec["fmt"]
    formatter = {k: kwargs[k] for k in tag_spec.keys() if k != "fmt"}

    _tag = _core_fmt.format(**formatter)

    if release_type in ["runtime", "cudnn"]:
        formatter["release_type"] = self._default_args["BENTOML_VERSION"]
        _tag = "-".join([_core_fmt.format(**formatter), release_type])

    # construct build_tag which is our base_image with $PYTHON_VERSION formats
    formatter.update({"python_version": "$PYTHON_VERSION", "release_type": "base"})
    _build_tag = _core_fmt.format(**formatter)

    return _tag, _build_tag


def aggregate_dists_releases(self, package: str) -> t.List[t.Tuple[str, str]]:
    """Aggregate all combos from manifest.yml"""
    return [(rel, d) for rel, dists in self.packages[package].items() for d in dists]


@lru_cache(maxsize=32)
def dockerfiles(self, pkg: str, distro: str, pyv: str) -> dict:
    """
    Generate template context for each distro releases.
    Args:
        pkg: release bentoml packages: bento-server
        distro: linux distro. eg: ubi
        pyv: python version
    Returns:
        build context that can be used with self.render
    """

    _cuda_comp: dict = get_data(self.cuda, self._cuda_version).copy()
    # TODO: Better type annotation for nested dict.
    _ctx: dict = self.releases[distro].copy()

    _ctx["package"] = pkg

    # setup envars
    _ctx["envars"] = self.envars()(_ctx["envars"])

    # set PYTHON_VERSION envars.
    set_data(_ctx, pyv, "envars", "PYTHON_VERSION")

    # setup cuda deps
    if _ctx["cuda_prefix_url"]:
        major, minor, _ = self._cuda_version.rsplit(".")
        _ctx["cuda"] = {
            "requires": _ctx["cuda_requires"],
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
        _ctx.pop("cuda_prefix_url")
        _ctx.pop("cuda_requires")
    else:
        logger.debug(f"CUDA is disabled for {distro}. Skipping...")
        for key in _ctx.copy():
            if DOCKERFILE_NVIDIA_REGEX.match(key):
                _ctx.pop(key)

    return _ctx


@lru_cache(maxsize=4)
def metadata(self, output_dir: str, pyv: str) -> t.Tuple[dict, dict]:
    """Setup templates context"""
    _paths: t.Dict[str, str] = {}
    _tags: dict = defaultdict()

    if os.geteuid() == 0:
        logger.warning(
            "Detected running as ROOT. Make sure not to "
            "use --generate dockerfiles while using "
            "manager_images. If you need to regenerate "
            "Dockerfiles use manager_dockerfiles instead."
        )

    for pkg in self.packages.keys():
        logger.info(f"Generate context for {pkg} python{pyv}")
        for (release, distro_version) in self.aggregate_dists_releases(pkg):
            logger.debug(f"Tag context for {release}:{distro_version}")

            # setup our build context
            _tag_ctx = self.dockerfiles(pkg, distro_version, pyv)
            template_dir = Path(_tag_ctx["templates_dir"])

            # generate our image tag.
            _release_tag, _build_tag = self.image_tags(
                python_version=pyv,
                release_type=release,
                suffixes=_tag_ctx["add_to_tags"],
            )

            # setup image tags.
            tag_keys = f"{_tag_ctx['package']}:{_release_tag}"

            # output path.
            base_output_path = Path(output_dir, pkg, f"{distro_version}")
            if release != "base":
                output_path = Path(base_output_path, release)
            else:
                output_path = base_output_path

            # input path.
            input_paths = [str(f) for f in walk(template_dir) if release in f.name]
            if "rhel" in template_dir.name:
                if "cudnn" in release:
                    input_paths = [
                        str(f)
                        for f in template_dir.iterdir()
                        if DOCKERFILE_NVIDIA_REGEX.match(f.name)
                    ]
                    _tag_ctx.update({"docker_build_path": str(output_path)})

            full_output_path = str(Path(output_path, DOCKERFILE_NAME))

            # # setup directory correspondingly, and add these paths
            set_data(_tags, _tag_ctx.copy(), tag_keys)
            set_data(_tags, input_paths, tag_keys, "input_paths")
            set_data(_tags, str(output_path), tag_keys, "output_path")
            if release != "base":
                set_data(_tags, _build_tag, tag_keys, "build_tags")

            # setup generated path with tag releases
            set_data(_paths, full_output_path, tag_keys)

    return _tags, _paths


def generate_readmes(self, output_dir: str, paths: t.Dict[str, str]) -> None:
    """
    Generate README output. We will also handle README context in memory
    Args:
        output_dir (:obj:`str`):
            target directory
        paths (:obj:`t.Dict[str, str]`):
            t.Dictionary of release tags and target dockerfile.
    """

    _release_context: t.MutableMapping = defaultdict(list)

    # check if given distro is supported per package.
    distros = [
        self.releases[release]["add_to_tags"] for release in self.releases.keys()
    ]
    for distro_version in distros:
        _release_context[distro_version] = sorted(
            [
                (release_tags, gen_path)
                for release_tags, gen_path in paths.items()
                if distro_version in release_tags and "base" not in release_tags
            ],
            key=operator.itemgetter(0),
        )
    _readme_context: t.Dict = {
        "ephemeral": False,
        "bentoml_package": "",
        "bentoml_release_version": FLAGS.bentoml_version,
        "release_info": _release_context,
    }

    for package in self.packages.keys():
        output_readme = Path("generated", package)
        set_data(_readme_context, package, "bentoml_package")
        self.render(
            input_path=README_TEMPLATE,
            output_path=output_readme,
            metadata=_readme_context,
        )

    # renders README for generated directory
    _readme_context["ephemeral"] = True
    if not Path(output_dir, "README.md").exists():
        self.render(
            input_path=README_TEMPLATE,
            output_path=Path(output_dir),
            metadata=_readme_context,
        )


def generate_dockerfiles(self) -> None:
    """
    Generate results targets from given context. This should only
    be used with --generate dockerfiles
    """

    if FLAGS.overwrite:
        # Consider generated directory ephemeral
        logger.info(f"Removing dockerfile dir: {FLAGS.dockerfile_dir}\n")
        shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
        Path(FLAGS.dockerfile_dir).mkdir(exist_ok=True)

    for tags_ctx in self._tags.values():
        for inp in tags_ctx["input_paths"]:
            build_tag = ""
            if "build_tags" in tags_ctx.keys():
                build_tag = tags_ctx["build_tags"]

            self.render(
                input_path=Path(inp),
                output_path=Path(tags_ctx["output_path"]),
                metadata=tags_ctx,
                build_tag=build_tag,
            )

    self.generate_readmes(FLAGS.dockerfile_dir, self._paths)


class BuildMixin(object):
    """Utilities for docker build process to be used as mixin."""

    def build_images(self) -> None:
        """Build images."""
        with ThreadPoolExecutor(max_workers=5) as executor:

            def _build_image(build_args):
                image_tag, dockerfile_path = build_args
                logger.info(f"Building {image_tag} from {dockerfile_path}")
                try:
                    # this is should be when there are changed in base image.
                    if FLAGS.overwrite:
                        docker_client.api.remove_image(image_tag, force=True)
                        # workaround for not having goto supports in Python.
                        raise ImageNotFound(f"Building {image_tag}")
                    if docker_client.api.history(image_tag):
                        logger.info(
                            f"Image {image_tag} is already built and --overwrite is not specified. Skipping..."
                        )
                        self._push_context[image_tag] = docker_client.images.get(
                            image_tag
                        )
                        return
                except ImageNotFound:
                    pyenv = get_data(self._tags, image_tag, "envars", "PYTHON_VERSION")
                    build_args = {"PYTHON_VERSION": pyenv}
                    # NOTES: https://warehouse.pypa.io/api-reference/index.html

                    logger.info(
                        f"Building docker image {image_tag} with {dockerfile_path}"
                    )
                    resp = docker_client.api.build(
                        timeout=FLAGS.timeout,
                        path=".",
                        nocache=False,
                        buildargs=build_args,
                        dockerfile=dockerfile_path,
                        tag=image_tag,
                        rm=True,
                    )
                    self.build_logs(resp, image_tag)

            base_tags, other_tags = self.build_tags()
            # Build all base images before starting other builds
            list(executor.map(_build_image, base_tags))
            list(executor.map(_build_image, other_tags))

    def build_tags(self) -> t.Tuple[t.Tuple[str, str], t.Tuple[str, str]]:
        """Generate build tags

        returns:
            (
                [(image_tag, dockerfile_path), ..],  # all base images
                [(image_tag, dockerfile_path), ..]  # all other images to build
            )
        """
        # We will build and push if args is parsed.
        # NOTE: type hint is a bit janky here as well
        if FLAGS.releases and (FLAGS.releases in DOCKERFILE_BUILD_HIERARCHY):
            return self.paths["base"].items(), self.paths[FLAGS.releases].items()
        else:
            # non "base" items
            items = []
            for release in DOCKERFILE_BUILD_HIERARCHY:
                if release != "base":
                    items += self.paths[release].items()

            return self.paths["base"].items(), items


class PushMixin(object):
    """Utilities for push tasks to be used as mixin."""

    def push_tags(self):  # type: ignore
        """Generate push tags"""
        _paths = deepcopy(self.paths)
        if FLAGS.releases and FLAGS.releases in DOCKERFILE_BUILD_HIERARCHY:
            _release_tag = _paths[FLAGS.releases]
        else:
            # we don't need to publish base tags.
            _paths.pop("base")
            _release_tag = flatten(
                map(lambda x: [_x for _x in x.keys()], _paths.values())
            )
        return _release_tag

    def push_images(self) -> None:
        """Push images to registries"""

        if not self._push_context:
            logger.debug("--generate images is not specified. Generate push context...")
            for image_tag, _ in self.build_tags()[1]:  # get non base image tags
                self._push_context[image_tag] = docker_client.images.get(image_tag)

        for registry, registry_spec in self.repository.items():
            # We will want to push README first.
            login_payload: t.Dict = {
                "username": os.getenv(registry_spec["user"]),
                "password": os.getenv(registry_spec["pwd"]),
            }
            api_url: str = get_nested(registry_spec, ["urls", "api"])

            for package, registry_url in registry_spec["registry"].items():
                _, _url = registry_url.split("/", maxsplit=1)
                readme_path = Path("generated", package, "README.md")
                repo_url: str = (
                    f"{get_nested(registry_spec, ['urls', 'repos'])}/{_url}/"
                )
                self.push_readmes(api_url, repo_url, readme_path, login_payload)

            if FLAGS.readmes:
                logger.info("--readmes is specified. Exit after pushing readmes.")
                return

            # Then push image to registry.
            with ThreadPoolExecutor(max_workers=5) as executor:
                for image_tag in self.push_tags():
                    image = self._push_context[image_tag]
                    reg, tag = image_tag.split(":")
                    registry = "".join(
                        [v for k, v in registry_spec["registry"].items() if reg in k]
                    )
                    logger.info(f"Uploading {image_tag} to {registry}")
                    future = executor.submit(
                        self.background_upload, image, tag, registry
                    )
                    logger.info(future.result)

    def push_readmes(
        self,
        api_url: str,
        repo_url: str,
        readme_path: Path,
        login_payload: t.Dict[str, str],
    ) -> None:
        """
        Push readmes to registries. Currently only supported for hub.docker.com

        Args:
            api_url (:obj:`str`):
                defined under ``manifest.yml``
            repo_url (:obj:`str`):
                defined under ``manifest.yml``
            readme_path (:obj:`pathlib.Path`):
                generated readmes path
            login_payload (:obj:`t.Dict[str, str]`):
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
                "X-CSRFToken": logins.cookies["csrftoken"],
            }
        )
        with readme_path.open("r") as f:
            # hub.docker.com specifics
            data = {"full_description": f.read()}
            self.patch(
                repo_url,
                data=json.dumps(data),
                headers=self.headers,
                cookies=logins.cookies,
            )

    def background_upload(self, image: "Image", tag: str, registry: str) -> None:
        """Upload a docker image."""
        image.tag(registry, tag=tag)
        logger.debug(f"Pushing {repr(image)} to {registry}...")
        resp = docker_client.images.push(registry, tag=tag, stream=True, decode=True)
        self.push_logs(resp, image.id)

    def auth_registries(self) -> None:
        """Setup auth registries for different registries"""

        # https://docs.docker.com/registry/spec/auth/token/
        pprint("Configure Docker")
        try:
            for registry, metadata in self.repository.items():
                if not os.getenv(metadata["pwd"]):
                    logger.warning(
                        f"If you are intending to use {registry}, make sure to set both "
                        f"{metadata['pwd']} and {metadata['user']} under {os.getcwd()}/.env. Skipping..."
                    )
                    continue
                else:
                    _user = os.getenv(metadata["user"])
                    _pwd = os.getenv(metadata["pwd"])

                    logger.info(
                        f"Logging into Docker registry {registry} with credentials..."
                    )
                    docker_client.api.login(
                        username=_user, password=_pwd, registry=registry
                    )
        except APIError:
            logger.info(
                "TLS handshake timeout. You can try resetting docker daemon."
                "If you are using systemd you can do `systemctl restart docker`"
            )
            docker_client.from_env(timeout=FLAGS.timeout)


class ManagerClient(Session, BuildMixin, PushMixin):
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

            contains mapping for docker.Images object from docker-py with correct
            releases type

    """  # noqa: LN001

    _paths: t.MutableMapping = defaultdict()
    _tags: t.MutableMapping = defaultdict()
    _push_context: dict = {}

    _template_env: Environment = Environment(
        extensions=["jinja2.ext.do"], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec: dict, cuda_version: str) -> None:
        super(ManagerClient, self).__init__()

        # We will also default bentoml_version to FLAGs.bentoml_version
        self._default_args: t.Dict = {"BENTOML_VERSION": FLAGS.bentoml_version}

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
            logger.debug("--python_version defined, ignoring SUPPORTED_PYTHON_VERSION")
            self.supported_python = FLAGS.python_version
            if self.supported_python not in SUPPORTED_PYTHON_VERSION:
                logger.critical(
                    f"{self.supported_python} is not supported. Allowed: {SUPPORTED_PYTHON_VERSION}"
                )
        else:
            self.supported_python = SUPPORTED_PYTHON_VERSION

        # generate Dockerfiles and README.
        pprint("Setting up metadata")
        for python_version in self.supported_python:
            _tags, _paths = self.metadata(FLAGS.dockerfile_dir, python_version)
            self._tags.update(_tags)
            self._paths.update(_paths)

        if FLAGS.dump_metadata:
            with open("./metadata.json", "w", encoding="utf-8") as ouf:
                ouf.write(json.dumps(self._tags, indent=2))
            ouf.close()

    @cached_property
    def paths(self) -> t.Dict[str, dict]:
        return {
            h: {k: self._paths[k] for k in self._paths.keys() if h in k}
            for h in DOCKERFILE_BUILD_HIERARCHY
        }

    def generate(self, dry_run: bool = False) -> None:
        if os.geteuid() != 0 and not dry_run:
            self.generate_dockerfiles()

    def build(self, dry_run: bool = False) -> None:
        if not dry_run:
            self.build_images()

    def push(self, dry_run: bool = False) -> None:
        # https://docs.docker.com/docker-hub/download-rate-limit/
        if not dry_run:
            self.auth_registries()
            self.push_images()


def setup_metadata(release_spec: "DictStrAny") -> "t.Tuple[DictStrAny, DictStrAny]":
    tags: "DictStrAny" = {}
    paths: "DictStrAny" = {}

    # namedtuple are slower than just having class objects since we are dealing
    # with a lot of nested properties.
    for key, values in release_spec.items():
        super().__setattr__(f"{key}", values)

    mapfunc(lambda x: list(flatten(x)), self.packages.values())

    # Workaround adding base to multistage enabled distros. This is very janky.
    for p in self.packages.values():
        p.update({"base": maxkeys(p)})

    # setup python version.
    if len(FLAGS.python_version) > 0:
        logger.debug("--python_version defined, ignoring SUPPORTED_PYTHON_VERSION")
        supported_python = FLAGS.python_version
        if supported_python not in SUPPORTED_PYTHON_VERSION:
            logger.critical(
                f"{supported_python} is not supported. Allowed: {SUPPORTED_PYTHON_VERSION}"
            )
    else:
        supported_python = SUPPORTED_PYTHON_VERSION

    # generate Dockerfiles and README.
    pprint("Setting up metadata")
    for python_version in supported_python:
        _tags, _paths = metadata(FLAGS.dockerfile_dir, python_version)
        tags.update(_tags)
        paths.update(_paths)

    if FLAGS.dump_metadata:
        with open("./metadata.json", "w", encoding="utf-8") as ouf:
            ouf.write(json.dumps(tags, indent=2))
    return tags, path


def build_images():
    ...


def authenticate_registries():
    ...


def push_images():
    ...


@graceful_exit
def main(argv: str) -> None:
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    # validate releases args.
    if FLAGS.releases and FLAGS.releases not in DOCKERFILE_BUILD_HIERARCHY:
        logger.critical(
            f"Invalid --releases arguments. Allowed: {DOCKERFILE_BUILD_HIERARCHY}"
        )

    # Parse specs from manifest.yml and validate it.
    pprint(f"Validating {FLAGS.manifest_file}")
    release_spec = load_manifest_yaml(FLAGS.manifest_file)
    if FLAGS.validate:
        # logger.info(json.dumps(release_spec['specs'], indent=2, sort_keys=True))
        logger.info(get_data(release_spec, "specs.repository"))
        # logger.info(release_spec['specs'])
        # logger.info(release_spec['repository'])
        # logger.info(release_spec['cuda'])
        # logger.info(release_spec['releases'])
        # logger.info(release_spec['packages'])
        return

    if not FLAGS.dry_run:
        # generate templates to correct directory.
        if "dockerfiles" in FLAGS.generate:
            if os.geteuid() != 0:
                generate_dockerfiles()
                # manager.generate(FLAGS.dry_run)
            else:
                logger.warning(
                    "Currently running as root to generate dockerfiles, this means dockerfile will have different permission. Exitting..."
                )
                sys.exit(1)
        elif "images" in FLAGS.generate:
            build_images()
            # manager.build(FLAGS.dry_run)

        # Push images when finish building.
        if FLAGS.push:
            authenticate_registries()
            push_images()
            # manager.push(dry_run=FLAGS.dry_run)


if __name__ == "__main__":
    app.run(main)
