from __future__ import annotations

import os
import json
import sys
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

from ._internal._funcs import send_log
from ._internal._funcs import stream_docker_logs
from ._internal._funcs import create_buildx_builder
from ._internal.groups import Environment
from ._internal.groups import pass_environment
from ._internal.exceptions import ManagerBuildFailed
from ._internal._configuration import DOCKERFILE_BUILD_HIERARCHY
from ._internal._configuration import DOCKER_TARGETARCH_LINUX_UNAME_ARCH_MAPPING

if TYPE_CHECKING:
    GenericDict = t.Dict[str, t.Any]

    Tags = t.Dict[str, t.Tuple[str, str, str, t.Dict[str, str], str, t.Tuple[str, ...]]]

logger = logging.getLogger(__name__)

BUILDER_LIST = []


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
        type=click.Choice(DOCKERFILE_BUILD_HIERARCHY),
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
        max_workers: int,
        dry_run: bool,
    ) -> None:
        """
        Build releases docker images with `buildx`. Utilize ThreadPoolExecutor.
        # of workers = max-workers * 2 to speed up releases since we have so many :)

        \b
        Usage:
            manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2
            manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2 --python-version 3.8 --python-version 3.9

        \b
        By default we will generate all given specs defined under manifest/<docker_package>.yml
        """

        base_tag, build_tag = order_build_hierarchy(ctx, releases)
        base_buildx_args = [i for i in buildx_args(ctx, base_tag)]
        build_buildx_args = [i for i in buildx_args(ctx, build_tag)]

        global BUILDER_LIST

        def build_multi_arch(cmd: "GenericDict") -> None:
            python_version = cmd["build_args"]["PYTHON_VERSION"]
            distro_name = cmd["labels"]["distro_name"]

            builder_name = f"{ctx.docker_package}-{python_version}-{distro_name}"

            try:
                builder = create_buildx_builder(name=builder_name)
                BUILDER_LIST.append(builder)
                send_log(f"Args: {cmd}")

                # NOTE: we need to push to registry when releasing different
                # * architecture.
                resp = docker.buildx.build(**cmd, builder=builder)
                stream_docker_logs(t.cast(t.Iterator[str], resp), cmd["tags"])
            except Exception as err:
                raise ManagerBuildFailed(f"Error while building:\n {err}") from err

        if dry_run:
            send_log("--dry-run, output tags to file.")
            with ctx._generated_dir.open(
                "base_tag.meta.json", "w"
            ) as f1, ctx._generated_dir.open("build_tag.meta.json", "w") as f2:
                json.dump(base_tag, f1)
                json.dump(build_tag, f2)
            return

        # We need to install QEMU to support multi-arch
        send_log(
            "[bold yellow]Installing binfmt to added support for QEMU...[/]",
            extra={"markup": True},
        )
        subprocess.check_call(
            args=["make", "-f", ctx._fs.getsyspath("Makefile"), "emulator"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(build_multi_arch, base_buildx_args))
                list(executor.map(build_multi_arch, build_buildx_args))
        except Exception as e:
            send_log(e, _manager_level=logging.ERROR)
            sys.exit(1)


def buildx_args(ctx: Environment, tags: Tags) -> t.Generator[GenericDict, None, None]:

    registry = os.environ.get("DOCKER_REGISTRY", None)
    if registry is None:
        raise ManagerBuildFailed("Failed to retrieve docker registry from envars.")

    for image_tag, tag_context in tags.items():
        (
            output_path,
            build_tag,
            python_version,
            labels,
            target_file,
            *platforms,
        ) = tag_context

        ref = f"{registry}/{image_tag}"
        # "cache_to": f"type=registry,ref={ref},mode=max",
        cache_from = [{"type": "registry", "ref": ref}]

        if build_tag != "":
            build_base_image = build_tag.replace("$PYTHON_VERSION", python_version)
            base_ref = f"{registry}/{build_base_image}"
            cache_from.append({"type": "registry", "ref": base_ref})

        yield {
            "context_path": ctx._fs.getsyspath("/"),
            "build_args": {"PYTHON_VERSION": python_version},
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
    ctx: Environment, releases: t.Optional[t.Iterable[str]]
) -> t.Tuple[Tags, Tags]:
    """
    Returns {tag: (docker_build_context_path, python_version, *platforms), ...} for base and other tags
    """

    release_context = ctx.release_ctx

    orders = {v: k for k, v in enumerate(DOCKERFILE_BUILD_HIERARCHY)}
    if releases and all(i in DOCKERFILE_BUILD_HIERARCHY for i in releases):
        target_releases = tuple(sorted(releases, key=lambda k: orders[k]))
    else:
        target_releases = DOCKERFILE_BUILD_HIERARCHY

    if "base" not in target_releases:
        target_releases = ("base",) + target_releases

    hierarchy = {
        f"{ctx.organization}/{tag}": (
            meta["output_path"],
            meta["build_tag"],
            cx.shared_context.python_version,
            {
                "distro_name": cx.shared_context.distro_name,
                "docker_package": cx.shared_context.docker_package,
                "maintainer": "BentoML Team <contact@bentoml.com>",
            },
            "Dockerfile-conda" if cx.shared_context.conda else "Dockerfile",
            *cx.shared_context.architectures,
        )
        for distro_contexts in release_context.values()
        for cx in distro_contexts
        for tag, meta in cx.release_tags.items()
        for tr in target_releases
        if tr in tag
    }

    base_tags = dicttoolz.keyfilter(lambda x: "base" in x, hierarchy)
    build_tags = dicttoolz.keyfilter(lambda x: "base" not in x, hierarchy)

    # non "base" items
    if ctx.distros:
        base_, build_ = {}, {}
        for distro in ctx.distros:
            base_.update(dicttoolz.keyfilter(lambda x: distro in x, base_tags))
            build_.update(dicttoolz.keyfilter(lambda x: distro in x, build_tags))
        return base_, build_
    else:
        return base_tags, build_tags
