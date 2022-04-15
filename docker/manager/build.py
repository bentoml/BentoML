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
import click
import fs.path
from toolz import dicttoolz
from python_on_whales import docker

from ._utils import send_log
from ._utils import stream_docker_logs
from ._utils import create_buildx_builder
from ._configuration import DOCKERFILE_BUILD_HIERARCHY

if TYPE_CHECKING:
    GenericDict = t.Dict[str, t.Any]

    Tags = t.Dict[str, t.Tuple[str, str, str, t.Dict[str, str], str, t.Tuple[str, ...]]]

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


def add_build_command(cli: click.Group) -> None:
    @cli.command()
    @click.option(
        "--releases",
        required=False,
        type=click.Choice(DOCKERFILE_BUILD_HIERARCHY + ("all",)),
        multiple=True,
        help=f"Targets releases for an image, default is to build all following the order: {DOCKERFILE_BUILD_HIERARCHY}",
    )
    @click.option(
        "--dry-run",
        required=False,
        is_flag=True,
        default=False,
        help="Dry-run",
    )
    @click.option(
        "--skip-base/--no-skip-base",
        required=False,
        is_flag=True,
        help="Skip building base releases",
    )
    @click.option(
        "--max-workers",
        required=False,
        type=int,
        default=5,
        help="Defauls with # of workers used for ThreadPoolExecutor",
    )
    @pass_environment
    def build(
        ctx: Environment,
        releases: t.Optional[t.Iterable[str]],
        skip_base: bool,
        max_workers: int,
        dry_run: bool,
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

        base_tag, base_tag = order_build_hierarchy(ctx, releases, skip_base=skip_base)
        base_buildx_args = [i for i in buildx_args(ctx, base_tag)]
        build_buildx_args = [i for i in buildx_args(ctx, base_tag)]

        global BUILDER_LIST

        def build_multi_arch(cmd: "GenericDict") -> None:
            tag = cmd["tags"]
            python_version = cmd["build_args"]["PYTHON_VERSION"]
            distro_name = cmd["labels"]["distro_name"]

            builder_name = f"{ctx.docker_package}-{python_version}-{distro_name}{'-conda' if 'conda' in tag else ''}"
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
            send_log("--dry-run, output tags to file.")
            with ctx._generated_dir.open(
                "base_tag.meta.json", "w"
            ) as f1, ctx._generated_dir.open("base_tag.meta.json", "w") as f2:
                json.dump(base_tag, f1)
                json.dump(base_tag, f2)
            return

        # We need to install QEMU to support multi-arch
        send_log(
            "[bold yellow]Installing binfmt to added support for QEMU...[/]",
            extra={"markup": True},
        )
        if os.path.exists("/.dockerenv"):
            path = "Makefile"
        else:
            path = ctx._fs.getsyspath("Makefile")
        subprocess.Popen(
            args=["make", "-f", path, "emulator"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        ).communicate()

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if not skip_base:
                    list(executor.map(build_multi_arch, base_buildx_args))
                list(executor.map(build_multi_arch, build_buildx_args))
        except Exception as e:
            send_log(e, _manager_level=logging.ERROR)
            sys.exit(1)


def buildx_args(ctx: Environment, tags: Tags) -> t.Generator[GenericDict, None, None]:

    registry = os.environ.get("DOCKER_REGISTRY", None)
    if registry is None:
        raise Exception("Failed to retrieve docker registry from envars.")

    for image_tag, tag_context in tags.items():
        (
            output_path,
            base_tag,
            python_version,
            labels,
            target_file,
            *platforms,
        ) = tag_context

        context_path = ctx._fs.getsyspath("/")
        if "cudnn" in image_tag:
            context_path = ctx._generated_dir.getsyspath(output_path)

        build_args = {"PYTHON_VERSION": python_version, "BUILDKIT_INLINE_CACHE": 1}
        if "ubi" in image_tag:
            build_args["UBIFORMAT"] = f'python-{python_version.replace(".", "")}'

        ref = f"{registry}/{ctx.organization}/{image_tag}"
        # "cache_to": f"type=registry,ref={ref},mode=max",
        cache_from = [{"type": "registry", "ref": ref}]

        if base_tag != "":
            build_base_image = base_tag.replace("$PYTHON_VERSION", python_version)
            base_ref = {
                "type": "registry",
                "ref": f"{registry}/{ctx.organization}/{build_base_image}",
            }
            cache_from.append(base_ref)
            # remove below after first releases
            if ctx.organization == "bentoml":
                prebuilt = {
                    "type": "registry",
                    "ref": f"{registry}/aarnphm/{build_base_image}",
                }
                cache_from.append(prebuilt)

        yield {
            "context_path": context_path,
            "build_args": build_args,
            "progress": "plain",
            "file": ctx._generated_dir.getsyspath(
                fs.path.combine(output_path, target_file)
            ),
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


def order_build_hierarchy(
    ctx: Environment,
    releases: t.Optional[t.Union[t.Literal["all"], t.Iterable[str]]],
    *,
    skip_base: bool = False,
) -> t.Tuple[Tags, Tags]:
    """
    Returns {tag: (docker_build_context_path, python_version, *platforms), ...} for base and other tags
    """

    release_context = ctx.release_ctx

    if releases == "all":
        target_releases = DOCKERFILE_BUILD_HIERARCHY
    else:
        if releases is not None and all(
            i in DOCKERFILE_BUILD_HIERARCHY for i in releases
        ):
            orders = {v: k for k, v in enumerate(DOCKERFILE_BUILD_HIERARCHY)}
            target_releases = tuple(sorted(releases, key=lambda k: orders[k]))
        else:
            target_releases = DOCKERFILE_BUILD_HIERARCHY

    if not skip_base and "base" not in target_releases:
        target_releases = ("base",) + target_releases

    hierarchy = {
        tag: (
            meta["output_path"],
            meta["base_tag"],
            cx.shared_context.python_version,
            {
                "distro_name": cx.shared_context.distro_name,
                "docker_package": cx.shared_context.docker_package,
                "maintainer": "BentoML Team <contact@bentoml.com>",
            },
            "Dockerfile-conda" if "conda" in tag else "Dockerfile",
            *cx.shared_context.architectures,
        )
        for distro_contexts in release_context.values()
        for cx in distro_contexts
        for tag, meta in cx.release_tags.items()
        for tr in target_releases
        if tr in tag
        and cx.shared_context.python_version not in cx.shared_context.ignore_python
    }

    base_tags = dicttoolz.keyfilter(lambda x: "base" in x, hierarchy)
    base_tags = dicttoolz.keyfilter(lambda x: "base" not in x, hierarchy)

    # non "base" items
    if ctx.distros:
        base_, build_ = {}, {}
        for distro in ctx.distros:
            base_.update(dicttoolz.keyfilter(lambda x: distro in x, base_tags))
            build_.update(dicttoolz.keyfilter(lambda x: distro in x, base_tags))
        return base_, build_
    else:
        return base_tags, base_tags
