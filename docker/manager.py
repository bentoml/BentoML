#!/usr/bin/env python3
import re
import os
import json
import typing as t
import operator
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
from utils import as_posix
from utils import get_data
from utils import set_data
from utils import shellcmd
from utils import graceful_exit
from utils import argv_validator
from utils import load_manifest_yaml
from utils import get_docker_platform_mapping
from utils import SUPPORTED_ARCHITECTURE_TYPE
from dotenv import load_dotenv
from jinja2 import Environment

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from utils import DictStrAny
    from utils import DictStrStr
    from utils import GenericFunc

    DictStrAnyDict = t.Dict[str, t.Dict[str, t.Any]]


# CONSTANTS

README_TEMPLATE = Path(os.getcwd(), "templates", "docs", "README.md.j2")

EXTENSION='.j2'
BASE_PREFIX='base-'
DOCKERFILE = "Dockerfile"
DOCKERFILE_TEMPLATE_SUFFIX = f"-{DOCKERFILE.lower()}{EXTENSION}"
DOCKERFILE_NVIDIA_REGEX = re.compile(r"(?:base-nvidia|base-cuda|cudnn)+")

if not os.path.exists("./.env"):
    logger.warning(f"Make sure to create .env file at {os.getcwd()}")
else:
    load_dotenv()
logger.warning(
    "Make sure to have https://github.com/christian-korneck/docker-pushrm installed since we will utilize this to push readme to docker repo."
)

LEGACY_FOLDER = "legacy"


def auth_registries() -> "DictStrStr":
    credentials = {}
    repository = get_data(release_spec, "repository")
    for repo, info in repository.items():
        _user = os.getenv(info["user"])
        _pwd = os.getenv(info["pwd"])
        if not _user:
            logger.warning(
                f"If you are intending to use {repo}, make sure to set both "
                f"{info['pwd']} and {info['user']} under {os.getcwd()}/.env. Skipping..."
            )
            _user = info["user"]
        if not _pwd:
            _pwd = info["pwd"]
        credentials[get_data(info, "registry", "bento-server")] = {
            "user": _user,
            "pwd": _pwd,
        }
    return credentials


# ┌──────────────────────────────────────────────────────────────────┐
# │ Context Generation                                               │
# │ tasks: handle processing manifest.yml and generate Jinja context │
# └──────────────────────────────────────────────────────────────────┘


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


def process_cuda_context(deps: "DictStrAny") -> "t.Tuple[DictStrAny, DictStrAny]":
    major, minor, patch = FLAGS.cuda_version.split(".")
    cuda_mapping = {
        "version": {
            "major": major,
            "minor": minor,
            "patch": patch,
            "shortened": f"{major}.{minor}",
        },
    }
    other_deps_mapping = {}
    for arch, requirements in deps.items():
        cuda_req = requirements.pop('cuda')
        components: "DictStrAnyDict" = cuda_req["components"]

        def cudnn_version():
            for k, v in components.items():
                if "cudnn" in k:
                    res = {"major_version": k[-1:], "version": v["version"]}
                    components.pop(k)
                    return res

        cuda_mapping[arch] = {
            "components": components,
            "cudnn": cudnn_version(),
            **cuda_req,
        }
        other_deps_mapping.update(requirements)
    return cuda_mapping, other_deps_mapping


def setup_base_context(
        python_version: str, distro: str, release_type: str, *, package: str, supported_arch_type: t.List[str]
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
    ctx["supported_arch_type"] = supported_arch_type
    return deepcopy(ctx)


def generation_metadata(ctx: "DictStrAny") -> "t.Tuple[DictStrAny, t.Tuple[str, DictStrAny]]":
    """
    Returns a tuple of two dictionaries:
        - contains image tags + build metadata
        - contains images tags + path to docker build file
    """
    releases_tag = ctx.pop("releases_tag")
    package = ctx.pop("package")
    release_type = ctx.pop("release_type")
    distro = ctx.pop("distro")
    supported_arch_type = ctx.pop("supported_arch_type")

    image_tag = f"{package}:{releases_tag}"

    architectures: t.List[str] = ctx.pop("architectures")
    # NOTE: for now its all CUDA deps
    dependencies: "DictStrAny" = ctx.pop("dependencies")

    # process input_paths and output_path
    base_output_path = Path(
        FLAGS.dockerfile_dir, package, distro
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
    ctx["docker_build_path"] = as_posix(output_path, DOCKERFILE)

    # EXPERIMENTAL: TensorRT + BentoML docker images
    trt_path = [str(f) for f in walk(template_dir) if "trt" in f.name]
    if len(trt_path) > 0:
        ctx["trt_input_path"] = trt_path[0]
        ctx["trt_output_path"] = as_posix(base_output_path, "Dockerfile-trt")

    # Setup CUDA deps and additional requirements.
    ctx["cuda"], ctx['additional_requirements'] = process_cuda_context(dependencies)

    # context for each platform build
    platform_mapping = get_docker_platform_mapping()
    ctx["cuda_arch_ctx"] = {
        arch: {
            "platform_args": f"--platform {platform_mapping[arch]}",
            "cuda_supported": arch in dependencies.keys(),
            "bentoml_supported": arch in supported_arch_type,
        }
        for arch in architectures
    }
    path_ctx = (image_tag, {"target_path": ctx['docker_build_path'], "supported_arch": [arch for arch in architectures if arch in supported_arch_type]})
    return {image_tag: ctx}, path_ctx 


def gen_ctx(*args: t.Any, **kwargs: t.Any) -> "t.Tuple[DictStrAny, t.List[t.Tuple[str ,DictStrAny]]]":

    supported_python_version: t.List[str] = kwargs.pop("supported_python_version")
    supported_arch_type: t.List[str] = kwargs.pop("supported_arch_type")
    build_ctxdata, path_ctxdata = {}, []

    # Create build context #
    for package in get_packages_spec():
        for python_version, (release_type, distro) in itertools.product(
            supported_python_version, aggregate_dist_releases(package)
        ):
            ctx = setup_base_context(
                python_version=python_version,
                distro=distro,
                release_type=release_type,
                package=package,
                supported_arch_type=supported_arch_type
            )
            build_ctx, path_ctx = generation_metadata(ctx)
            build_ctxdata.update(build_ctx)
            path_ctxdata.append(path_ctx)

    if FLAGS.dump_metadata:
        with open("./metadata.json", "w", encoding="utf-8") as ouf:
            ouf.write(json.dumps(build_ctxdata, indent=2))
    return build_ctxdata, path_ctxdata


# ┌─────────────────────────────────────────────────┐
# │ Dockerfile Generation                           │
# │ tasks: generate Dockerfile from Jinja templates │
# └─────────────────────────────────────────────────┘


def render(
    input_path: Path,
    output_path: Path,
    metadata: "DictStrAny",
    arch:t.Optional[str]=None,
    build_tag: t.Optional[str] = None,
):
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
    template_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"], trim_blocks=True, lstrip_blocks=True
    )
    input_name = input_path.name
    if "readme".upper() in input_name:
        output_name = input_path.stem
    elif 'dockerfile' in input_name:
        output_name = DOCKERFILE
    else:
        # specific cases for rhel
        output_name = input_name[len(BASE_PREFIX):-len(EXTENSION)] if input_name.startswith(BASE_PREFIX) and input_name.endswith(EXTENSION) else input_name
        if arch:
            output_name += f"-{arch}"
    if not output_path.exists():
        logger.debug(f"Creating {as_posix(output_path)}...")
        output_path.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as inf:
        template = template_env.from_string(inf.read())

    with output_path.joinpath(output_name).open("w") as ouf:
        ouf.write(template.render(metadata=metadata, build_tag=build_tag))


def generate_readmes(
        output_dir: str, package: str, path_ctx: "t.List[t.Tuple[str, DictStrAny]]"
) -> None:
    distros = get_packages_spec("bento-server", "base")
    release_context = {}
    for distro_version in distros:
        release_context[distro_version] = {
                "image_tag": sorted(
            [
                (image_tag, tag_info['target_path'])
                for (image_tag, tag_info) in [tpl for tpl in path_ctx if distro_version in tpl[0]]
                if "base" not in image_tag 
            ],
            key=operator.itemgetter(0),
            ), "supported_arch": list(set([i for tag_info in [tpl[1] for tpl in path_ctx if distro_version in tpl[0]] for i in tag_info['supported_arch']]))
                }
    readme_context = {
        "ephemeral": False,
        "bentoml_package": package,
        "bentoml_release_version": FLAGS.bentoml_version,
        "release_info": release_context,
    }
    output_readme = Path(output_dir, package)
    render(
        input_path=README_TEMPLATE, output_path=output_readme, metadata=readme_context
    )

    readme_context["ephemeral"] = True
    render(
        input_path=README_TEMPLATE,
        output_path=Path(output_dir),
        metadata=readme_context,
    )


def generate_dockerfiles():
    ...


def build_images():
    ...


def authenticate_registries():
    ...


def push_images():
    ...


@graceful_exit
@argv_validator(length=1)
def main(*args: t.Any, **kwargs: t.Any) -> None:

    global release_spec, get_architecture_spec, get_packages_spec
    release_spec = load_manifest_yaml(FLAGS.manifest_file, FLAGS.validate)
    get_architecture_spec = partial(
        get_data, release_spec, "releases", f"w_cuda_v{FLAGS.cuda_version}"
    )
    get_packages_spec = partial(get_data, release_spec, "packages")

    # auth cred to pass to skopeo
    credentials = auth_registries()

    # generate context
    build_ctx, path_ctx = gen_ctx(*args, **kwargs)

    # generate jinja templates
    for package in get_packages_spec():
        generate_readmes(output_dir=FLAGS.dockerfile_dir, package=package, path_ctx=path_ctx)


if __name__ == "__main__":
    app.run(main)
