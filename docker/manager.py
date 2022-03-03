#!/usr/bin/env python3
import os
import re
import sys
import json
import typing as t
from typing import TYPE_CHECKING
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from log import logger
from absl import app
from utils import FLAGS
from utils import jprint
from utils import pprint
from utils import flatten
from utils import mapfunc
from utils import maxkeys
from utils import get_data
from utils import set_data
from utils import shellcmd
from utils import graceful_exit
from utils import argv_validator
from utils import cached_property
from utils import DOCKERFILE_NAME
from utils import NVIDIA_REPO_URL
from utils import README_TEMPLATE
from utils import load_manifest_yaml
from utils import NVIDIA_ML_REPO_URL
from utils import DOCKERFILE_NVIDIA_REGEX
from utils import SUPPORTED_PYTHON_VERSION
from utils import DOCKERFILE_TEMPLATE_SUFFIX
from utils import SUPPORTED_ARCHITECTURE_TYPE
from dotenv import load_dotenv
from jinja2 import Environment
from requests import Session

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from utils import DictStrAny
    from utils import GenericFunc

    DictStrStr = t.Dict[str, str]

if not os.path.exists("./.env"):
    logger.warning(f"Make sure to create .env file at {os.getcwd()}")
else:
    load_dotenv()
logger.warning(
    "Make sure to have https://github.com/christian-korneck/docker-pushrm installed since we will utilize this to push readme to docker repo."
)


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


def generate_dockerfiles():
    ...


def build_images():
    ...


def authenticate_registries():
    ...


def push_images():
    ...


def process_envars(additional_args: "t.Optional[DictStrStr]" = None) -> "GenericFunc":
    args = {"BENTOML_VERSION": FLAGS.bentoml_version}
    if additional_args is None:
        additional_args = {}
    args.update(additional_args)

    def update_args_dict(
        updater: "t.Union[str, DictStrAny, t.List[str]]",
    ) -> "DictStrAny":
        # Args will have format ARG=foobar
        if isinstance(updater, str):
            arg, _, value = updater.partition("=")
            args[arg] = value
        elif isinstance(updater, list):
            for v in updater:
                update_args_dict(v)
        elif isinstance(updater, dict):
            for arg, value in updater.items():
                args[arg] = value
        else:
            logger.error(f"cannot add to args dict with unknown type {type(updater)}")
        return args

    return update_args_dict


def generate_image_tags(**kwargs: str) -> t.Tuple[str, str]:
    """Return a base and a build tag for a given images based on release type"""
    release_type = kwargs.get("release_type", "")
    tag_spec = get_data(release_spec, "specs", "tag")
    core_tag_fmt: str = tag_spec.pop("fmt")
    formatter = {k: kwargs[k] for k in tag_spec}

    if release_type != "devel":
        formatter["release_type"] = FLAGS.bentoml_version
        tag = "-".join([core_tag_fmt.format(**formatter), release_type])
    else:
        tag = core_tag_fmt.format(**formatter)

    # construct build_tag which is our base_image with $PYTHON_VERSION formats
    formatter.update({"python_version": "$PYTHON_VERSION", "release_type": "base"})
    build_tag = core_tag_fmt.format(**formatter)

    return tag, build_tag


def aggregate_dist_releases() -> t.List[t.Tuple[str, str]]:
    return [(rel, d) for rel, dist in get_bento_server_spec().items() for d in dist]


@graceful_exit
@argv_validator(length=1)
def main(*args: t.Any, **kwargs: t.Any) -> None:
    global release_spec, get_architecture_spec, get_bento_server_spec
    release_spec = load_manifest_yaml(FLAGS.manifest_file, FLAGS.validate)
    get_architecture_spec = partial(
        get_data, release_spec, "releases", f"w_cuda_v{FLAGS.cuda_version}"
    )
    get_bento_server_spec = partial(get_data, release_spec, "packages", "bento-server")

    supported_python_version: t.List[str] = kwargs.pop("supported_python_version")

    print(get_architecture_spec("debian11").keys())
    if FLAGS.validate:
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
