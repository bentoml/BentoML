#!/usr/bin/env python3
import os
import json
import typing as t
import itertools
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from log import logger
from absl import app
from utils import walk
from utils import FLAGS
from utils import jprint
from utils import get_data
from utils import set_data
from utils import shellcmd
from utils import graceful_exit
from utils import argv_validator
from utils import DOCKERFILE_NAME
from utils import README_TEMPLATE
from utils import load_manifest_yaml
from utils import DOCKERFILE_NVIDIA_REGEX
from utils import DOCKERFILE_TEMPLATE_SUFFIX
from utils import get_docker_platform_mapping
from dotenv import load_dotenv
from jinja2 import Environment

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from utils import DictStrAny
    from utils import DictStrStr
    from utils import GenericFunc

if not os.path.exists("./.env"):
    logger.warning(f"Make sure to create .env file at {os.getcwd()}")
else:
    load_dotenv()
logger.warning(
    "Make sure to have https://github.com/christian-korneck/docker-pushrm installed since we will utilize this to push readme to docker repo."
)

LEGACY_FOLDER = "legacy"
PACKAGE = "bento-server"


def generate_dockerfiles():
    ...


def build_images():
    ...


def authenticate_registries():
    ...


def push_images():
    ...


def aggregate_dist_releases(package: str) -> t.List[t.Tuple[str, str]]:
    " " " Returns a list of mapping between release_type and distros " " "
    return [(rel, d) for rel, dist in get_packages_spec(package).items() for d in dist]


def process_envars(
    additional_args: "t.Optional[DictStrStr]" = None,
) -> "GenericFunc[DictStrAny]":
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
    core_tag_fmt: str = tag_spec["fmt"]
    formatter = {k: kwargs[k] for k in tag_spec if k != "fmt"}

    if release_type in ["runtime", "cudnn"]:
        formatter["release_type"] = FLAGS.bentoml_version
        tag = "-".join([core_tag_fmt.format(**formatter), release_type])
    else:
        tag = core_tag_fmt.format(**formatter)

    # construct build_tag which is our base_image with $PYTHON_VERSION formats
    formatter.update({"python_version": "$PYTHON_VERSION", "release_type": "base"})
    build_tag = core_tag_fmt.format(**formatter)

    return tag, build_tag


def process_cuda_context(requirements: "DictStrAny") -> "DictStrAny":
    components: "DictStrStr" = requirements["components"]

    def cudnn_version():
        for k, v in components.items():
            if "cudnn" in k:
                res = {"major_version": k[-1:], "version": v}
                components.pop(k)
                return res

    major, minor, patch = FLAGS.cuda_version.split(".")
    return {
        "components": components,
        "cudnn": cudnn_version(),
        **requirements,
        "version": {
            "major": major,
            "minor": minor,
            "patch": patch,
            "shortened": f"{major}.{minor}",
        },
    }


def setup_base_context(
    python_version: str, distro: str, release_type: str, *, package: str
) -> "DictStrAny":
    ctx = get_architecture_spec(distro)

    # Setup base context
    releases_tag, build_tag = generate_image_tags(
        python_version=python_version,
        release_type=release_type,
        suffixes=ctx["suffixes"],
    )
    if release_type != "base":
        set_data(ctx, build_tag, "build_tag")
    set_data(ctx, package, "package")
    set_data(
        ctx, process_envars({"PYTHON_VERSION": python_version})(ctx["envars"]), "envars"
    )
    ctx["releases_tag"] = releases_tag
    ctx["distro"] = distro
    ctx["release_type"] = release_type
    return deepcopy(ctx)


def generation_metadata(ctx: "DictStrAny") -> "t.Tuple[DictStrAny, DictStrAny]":
    """
    Returns a tuple of two dictionaries:
        - contains image tags + build metadata
        - contains images tags + path to docker build file
    """
    tags_metadata = {}
    releases_tag = ctx.pop("releases_tag")
    package = ctx.pop("package")
    release_type = ctx.pop("release_type")
    distro = ctx.pop("distro")
    docker_image_tags = f"{package}:{releases_tag}"

    supported_arch: t.List[str] = ctx.pop("architectures")
    # NOTE: for now its all CUDA deps
    dependencies: "DictStrAny" = ctx.pop("dependencies")

    # process input_paths and output_path
    base_output_path = Path(
        FLAGS.dockerfile_dir, package, FLAGS.bentoml_version, distro
    )
    output_path = Path(base_output_path, release_type)

    # input paths
    template_dir = Path(os.path.dirname(__file__), ctx["templates_dir"])
    if "rhel" in template_dir.name and "cudnn" in release_type:
        input_paths = [
            str(f)
            for f in template_dir.iterdir()
            if DOCKERFILE_NVIDIA_REGEX.match(f.name)
        ]
    else:
        input_paths = [
            str(f)
            for f in walk(template_dir, exclude=(LEGACY_FOLDER,))
            if release_type in f.name
        ]

    ctx["input_paths"] = input_paths
    ctx["docker_build_path"] = Path(output_path, DOCKERFILE_NAME).as_posix()

    # EXPERIMENTAL: TensorRT + BentoML docker images
    trt_path = [str(f) for f in walk(template_dir) if "trt" in f.name]
    if len(trt_path) > 0:
        ctx["trt_input_path"] = trt_path[0]
        ctx["trt_output_path"] = Path(base_output_path, "trt.Dockerfile").as_posix()

    # context for each platform build
    platform_mapping = get_docker_platform_mapping()
    for arch in supported_arch:
        tags_metadata[arch] = {"platform_args": platform_mapping[arch]}
        _ctx = deepcopy(ctx)
        if arch in dependencies:
            _ctx["cuda"] = {}
            set_data(_ctx, process_cuda_context(dependencies[arch]), "cuda")
        tags_metadata[arch].update(_ctx)
    return {docker_image_tags: tags_metadata}, {
        docker_image_tags: Path(output_path, DOCKERFILE_NAME).as_posix()
    }


def gen_ctx(*args: t.Any, **kwargs: t.Any) -> "t.Tuple[DictStrAny, DictStrAny]":

    supported_python_version: t.List[str] = kwargs.pop("supported_python_version")
    build_metadata, path_metadata = {}, {}

    # Create build context #
    for package in get_packages_spec():
        for python_version, (release_type, distro) in itertools.product(
            supported_python_version, aggregate_dist_releases(package)
        ):
            ctx = setup_base_context(
                python_version=python_version,
                distro=distro,
                release_type=release_type,
                package=PACKAGE,
            )
            build_ctx, path_ctx = generation_metadata(ctx)
            build_metadata.update(build_ctx)
            path_metadata.update(path_ctx)

    if FLAGS.dump_metadata:
        with open("./metadata.json", "w", encoding="utf-8") as ouf:
            ouf.write(json.dumps(build_metadata, indent=2))
    return build_metadata, path_metadata


@graceful_exit
@argv_validator(length=1)
def main(*args: t.Any, **kwargs: t.Any) -> None:

    global release_spec, get_architecture_spec, get_packages_spec
    release_spec = load_manifest_yaml(FLAGS.manifest_file, FLAGS.validate)
    get_architecture_spec = partial(
        get_data, release_spec, "releases", f"w_cuda_v{FLAGS.cuda_version}"
    )
    get_packages_spec = partial(get_data, release_spec, "packages")

    # generate context
    build_meta, path_meta = gen_ctx(*args, **kwargs)
    # generate jinja templates


if __name__ == "__main__":
    app.run(main)
