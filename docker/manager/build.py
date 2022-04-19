from __future__ import annotations

import os
import sys
import json
import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

import fs
import fs.path
from toolz import dicttoolz
from simple_di import inject
from simple_di import Provide
from manager._make import DistrosManifest
from manager._make import generate_base_tags
from manager._make import generate_releases_tags_mapping
from manager._utils import send_log
from manager._utils import stream_docker_logs
from manager._utils import create_buildx_builder
from python_on_whales import docker
from manager._configuration import get_manifest_info
from manager._configuration import DockerManagerContainer

if TYPE_CHECKING:
    GenericDict = t.Dict[str, t.Any]

    Tags = t.Dict[str, t.Tuple[str, str, str, t.Dict[str, str], str, t.Tuple[str, ...]]]
    from fs.base import FS

logger = logging.getLogger(__name__)

BUILDER_LIST = []

# We only care about linux mapping to uname
DOCKER_TARGETARCH_LINUX_UNAME_ARCH_MAPPING = {
    "amd64": "x86_64",
    "arm64v8": "aarch64",
    "arm32v5": "armv7l",
    "arm32v6": "armv7l",
    "arm32v7": "armv7l",
    "i386": "386",
    "ppc64le": "ppc64le",
    "s390x": "s390x",
    "riscv64": "riscv64",
    "mips64le": "mips64",
}


def process_docker_arch(arch: str) -> str:
    # NOTE: hard code these platform, seems easy to manage
    if "arm64" in arch:
        fmt = "arm64/v8"
    elif "arm32" in arch:
        fmt = "/".join(arch.split("32"))
    elif "i386" in arch:
        fmt = arch[1:]
    else:
        fmt = arch
    return fmt


def get_docker_platform_mapping() -> t.Dict[str, str]:
    return {
        k: f"linux/{process_docker_arch(k)}"
        for k in DOCKER_TARGETARCH_LINUX_UNAME_ARCH_MAPPING
    }


@inject
def order_build_hierarchy(
    distro_name: str,
    ctx: DistrosManifest,
    release_types: t.Union[t.Literal["all"], t.Iterable[str]] = "all",
    *,
    docker_package: str = DockerManagerContainer.docker_package,
) -> t.Tuple[Tags, Tags]:
    """
    Returns {tag: (docker_build_context_path, python_version, *platforms), ...} for base and other tags
    """

    release_hierarchy = DockerManagerContainer.RELEASE_TYPE_HIERARCHY

    if release_types == "all":
        target_releases = release_hierarchy
    else:
        if not all(i in release_hierarchy for i in release_types):
            raise ValueError(
                f"Invalid release type: {release_types}. Valid release types are: {release_hierarchy}"
            )
        orders = {v: k for k, v in enumerate(release_hierarchy)}
        target_releases = tuple(sorted(release_types, key=lambda k: orders[k]))

    if "base" not in target_releases:
        target_releases = ("base",) + target_releases

    base_tags = generate_base_tags(ctx.suffixes, ctx.conda)

    hierarchy = {
        tag: (
            fs.path.split(output_path)[0],  # output_directory
            base_tags["conda"] if "conda" in tag else base_tags["base"],  # base tags
            tag.split(":")[-1].split("-")[1].strip("python"),  # python version
            {
                "distro_name": distro_name,
                "docker_package": docker_package,
                "maintainer": "BentoML Team <contact@bentoml.com>",
            },
            "Dockerfile-conda" if "conda" in tag else "Dockerfile",
            *ctx.architectures,
        )
        for tag, output_path in generate_releases_tags_mapping(
            distro_name, ctx, skip_base_image=False
        )
        for tr in target_releases
        if tr in tag
    }

    base_tags = dicttoolz.keyfilter(lambda x: "base" in x, hierarchy)
    build_tags = dicttoolz.keyfilter(lambda x: "base" not in x, hierarchy)

    return base_tags, build_tags


@inject
def buildx_args(
    tags: Tags,
    registry: str = "docker.io",
    *,
    _fs: FS = Provide[DockerManagerContainer.root_fs],
    _generated_fs: FS = Provide[DockerManagerContainer.generated_fs],
    organization: str = Provide[DockerManagerContainer.organization],
) -> t.Generator[GenericDict, None, None]:

    for image_tag, tag_context in tags.items():
        (
            output_path,
            base_tag,
            python_version,
            labels,
            target_file,
            *platforms,
        ) = tag_context

        context_path = _fs.getsyspath("/")
        if "cudnn" in image_tag:
            context_path = _generated_fs.getsyspath(output_path)

        build_args = {"PYTHON_VERSION": python_version, "BUILDKIT_INLINE_CACHE": 1}
        if "ubi" in image_tag:
            build_args["UBIFORMAT"] = f'python-{python_version.replace(".", "")}'

        ref = f"{registry}/{organization}/{image_tag}"
        # "cache_to": f"type=registry,ref={ref},mode=max",
        cache_from = [{"type": "registry", "ref": ref}]

        if base_tag != "" and "base" not in image_tag:
            build_base_image = base_tag.replace("$PYTHON_VERSION", python_version)
            base_ref = {
                "type": "registry",
                "ref": f"{registry}/{organization}/{build_base_image}",
            }
            cache_from.append(base_ref)
            # TODO: remove below after first releases
            if organization == "bentoml":
                prebuilt = {
                    "type": "registry",
                    "ref": f"{registry}/aarnphm/{build_base_image}",
                }
                cache_from.append(prebuilt)

        yield {
            "context_path": context_path,
            "build_args": build_args,
            "progress": "plain",
            "file": _generated_fs.getsyspath(fs.path.combine(output_path, target_file)),
            "platforms": [
                v for k, v in get_docker_platform_mapping().items() if k in platforms
            ],
            "labels": labels,
            "tags": ref,
            "push": True,
            "pull": True,
            "cache": True,
            "stream_logs": True,
            "cache_from": cache_from,
        }


@inject
def build_images(
    releases: t.Iterable[str],
    distros: t.Iterable[str],
    dry_run: bool,
    max_workers: int = 5,
    *,
    _fs: FS = Provide[DockerManagerContainer.root_fs],
    _generated_fs: FS = Provide[DockerManagerContainer.generated_fs],
    docker_package: str = DockerManagerContainer.docker_package,
) -> None:
    """
    Build releases docker images with `buildx`. Utilize ThreadPoolExecutor.
    # of workers = max-workers to speed up releases since we have so many :)

    \b
    Usage:
        manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2
        manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2 --python-version 3.8 --python-version 3.9

    \b
    By default we will generate all given specs defined under manifest/<docker_package>.yml
    """

    bases, builds = {}, {}

    if not all(d in DockerManagerContainer.SUPPORTED_DISTRO_TYPE for d in distros):
        raise ValueError(
            f"Invalid distro type: {distros}. Valid distro types are: {DockerManagerContainer.SUPPORTED_DISTRO_TYPE}"
        )

    for distro, context in get_manifest_info().items():
        if distro not in distros:
            continue
        base, build = order_build_hierarchy(
            distro, DistrosManifest(**context), release_types=releases
        )
        bases.update(base)
        builds.update(build)

    base_buildx_args = [i for i in buildx_args(bases)]
    build_buildx_args = [i for i in buildx_args(builds)]

    global BUILDER_LIST

    def build_multi_arch(cmd: GenericDict) -> None:
        tag = cmd["tags"]
        python_version = cmd["build_args"]["PYTHON_VERSION"]
        distro_name = cmd["labels"]["distro_name"]

        builder_name = f"{docker_package}-{python_version}-{distro_name}{'-conda' if 'conda' in tag else ''}"
        send_log(f"args for buildx: {cmd}")

        try:
            builder = create_buildx_builder(name=builder_name)
            BUILDER_LIST.append(builder)

            # NOTE: we need to push to registry when releasing different
            # * architecture.
            resp = docker.buildx.build(**cmd, builder=builder)
            stream_docker_logs(t.cast(t.Iterator[str], resp), tag)
        except Exception as err:
            logger.error(f"Error while building:\n {err}")
            sys.exit(1)

    if dry_run:
        send_log("--dry-run, output tags to generated/{bases,builds}.meta.json ...")
        with _generated_fs.open("bases.meta.json", "w") as f1, _generated_fs.open(
            "builds.meta.json", "w"
        ) as f2:
            json.dump(bases, f1)
            json.dump(builds, f2)
        return

    # We need to install QEMU to support multi-arch
    send_log(
        "[bold yellow]Installing binfmt to added support for QEMU...[/]",
        extra={"markup": True},
    )

    if os.path.exists("/.dockerenv"):
        path = "Makefile"
    else:
        path = _fs.getsyspath("Makefile")

    subprocess.Popen(
        args=["make", "-f", path, "emulator"],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ).communicate()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(build_multi_arch, base_buildx_args))
            list(executor.map(build_multi_arch, build_buildx_args))
    except Exception as e:
        send_log(e, _manager_level=logging.ERROR)
        sys.exit(1)


def flatten_list(lst: t.List[t.List[str]]) -> t.List[str]:
    return [item for sublist in lst for item in sublist]


def main(args):
    if args.releases is None:
        releases = DockerManagerContainer.RELEASE_TYPE_HIERARCHY
    else:
        releases = flatten_list(args.releases)

    if args.distros is None:
        distros = DockerManagerContainer.SUPPORTED_DISTRO_TYPE
    else:
        distros = flatten_list(args.distros)

    if args.dry_run:
        send_log(
            f"input args: --releases {releases} --dry-run {args.dry_run} --max-worker {args.max_worker} --bentoml-version {args.bentoml_version} --distros {distros}\n",
        )

    build_images(
        releases=releases,
        distros=distros,
        dry_run=args.dry_run,
        max_workers=args.max_worker,
    )
