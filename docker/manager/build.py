import os
import json
import typing as t
import logging
import traceback
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

import fs
import yaml
import click
import fs.path
from toolz import dicttoolz
from manager._utils import run
from manager._utils import stream_logs
from manager._utils import DOCKERFILE_NAME
from manager._utils import create_buildx_builder
from manager._utils import DOCKERFILE_BUILD_HIERARCHY
from manager._utils import get_docker_platform_mapping
from python_on_whales import docker
from manager._exceptions import ManagerBuildFailed
from manager._click_utils import Environment
from manager._click_utils import pass_environment

if TYPE_CHECKING:
    from manager._types import GenericDict

    Tags = t.Dict[str, t.Tuple[str, str, t.Dict[str, str], t.Tuple[str, ...]]]

logger = logging.getLogger(__name__)

BUILDER_LIST = []
BUILT_IMAGE = []


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
            manager build --bentoml-version 1.0.0a5 --registry ecr
            manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2
            manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2 --python-version 3.8 --python-version 3.9
            manager build --bentoml-version 1.0.0a5 --docker-package <other-package>

        \b
        By default we will generate all given specs defined under manifest/<docker_package>.yml
        """

        base_tag, build_tag = order_build_hierarchy(ctx, releases)
        base_buildx_args = [i for i in buildx_args(ctx, base_tag)]
        build_buildx_args = [i for i in buildx_args(ctx, build_tag)]

        if ctx.xx_image == "local-xx":
            prepare_xx_image(ctx)

        built_img_metafile = "built_image.meta.yaml"
        global BUILDER_LIST, BUILT_IMAGE

        if ctx._generated_dir.exists(built_img_metafile):
            with ctx._generated_dir.open(built_img_metafile, "r") as f:
                BUILT_IMAGE = yaml.safe_load(f)

        def build_multi_arch(cmd: "GenericDict") -> None:
            python_version = cmd["build_args"]["PYTHON_VERSION"]
            distro_name = cmd["labels"]["distro_name"]

            builder_name = f"{ctx.docker_package}-{python_version}-{distro_name}"

            if cmd["tags"] in BUILT_IMAGE:
                logger.info(f"{cmd['tags']} is already built and pushed.")
                return
            try:
                builder = create_buildx_builder(name=builder_name)
                BUILDER_LIST.append(builder)
                if ctx.verbose:
                    logger.info(f"Args: {cmd}")

                # NOTE: we need to push to registry when releasing different
                # * architecture.
                resp = docker.buildx.build(**cmd, builder=builder)
                stream_logs(t.cast(t.Iterator[str], resp), cmd["tags"])
            except Exception as e:
                traceback.format_exc()
                raise ManagerBuildFailed(f"Error while building:\n {e}") from e
            finally:
                BUILT_IMAGE.append(cmd["tags"])

        if dry_run:
            logger.info("--dry-run, output tags to file.")
            with ctx._generated_dir.open(
                "base_tag.meta.json", "w"
            ) as f1, ctx._generated_dir.open("build_tag.meta.json", "w") as f2:
                json.dump(base_tag, f1)
                json.dump(build_tag, f2)
            return

        # We need to install QEMU to support multi-arch
        logger.info(
            "[bold yellow]Installing binfmt to added support for QEMU...[/]",
            extra={"markup": True},
        )
        _ = run("make", "emulator", "-f", ctx._fs.getsyspath("Makefile"))

        try:
            with ThreadPoolExecutor(max_workers=max_workers * 2) as executor:
                list(executor.map(build_multi_arch, base_buildx_args))
                list(executor.map(build_multi_arch, build_buildx_args))
        except Exception as e:
            logger.error(e)
            traceback.format_exc()
            raise
        finally:
            with ctx._manifest_dir.open(built_img_metafile, "w", encoding="utf-8") as f:
                yaml.dump(BUILT_IMAGE, f)


def buildx_args(
    ctx: Environment, tags: "Tags"
) -> "t.Generator[GenericDict, None, None]":

    REGISTRIES_ENVARS_MAPPING = {"docker.io": "DOCKER_URL", "ecr": "AWS_URL"}

    if ctx.push_registry is None:
        tag_prefix = None
    else:
        tag_prefix = os.environ.get(REGISTRIES_ENVARS_MAPPING[ctx.push_registry], None)
        if tag_prefix is None:
            raise ManagerBuildFailed(
                "Failed to retrieve URL prefix for docker registry."
            )

    for image_tag, tag_context in tags.items():
        output_path, python_version, labels, *platforms = tag_context

        ref = image_tag if tag_prefix is None else f"{tag_prefix}/{image_tag}"

        yield {
            "context_path": ctx._fs.getsyspath("/"),
            "build_args": {"PYTHON_VERSION": python_version},
            "progress": "plain",
            "file": ctx._generated_dir.getsyspath(
                fs.path.combine(output_path, DOCKERFILE_NAME)
            ),
            "platforms": [
                v for k, v in get_docker_platform_mapping().items() if k in platforms
            ],
            "labels": labels,
            "tags": ref,
            "push": True,
            "pull": True,
            "stream_logs": True,
            "cache_from": f"type=registry,ref={ref}",
            "cache_to": f"type=registry,ref={ref},mode=max",
        }


def order_build_hierarchy(
    env: Environment, releases: t.Optional[t.Iterable[str]]
) -> "t.Tuple[Tags, Tags]":
    """
    Returns {tag: (docker_build_context_path, python_version, *platforms), ...} for base and other tags
    """

    def filter_support_architecture(
        python_version: str, arch: t.List[str]
    ) -> t.List[str]:
        return arch if python_version != "3.6" else ["amd64"]

    release_context = env.release_ctx

    orders = {v: k for k, v in enumerate(DOCKERFILE_BUILD_HIERARCHY)}
    if releases and all(i in DOCKERFILE_BUILD_HIERARCHY for i in releases):
        target_releases = tuple(sorted(releases, key=lambda k: orders[k]))
    else:
        target_releases = DOCKERFILE_BUILD_HIERARCHY

    if "base" not in target_releases:
        target_releases = ("base",) + target_releases

    hierarchy = {
        tag: (
            meta["output_path"],
            ctx.shared_ctx.python_version,
            {
                "distro_name": ctx.shared_ctx.distro_name,
                "docker_package": ctx.shared_ctx.docker_package,
            },
            *filter_support_architecture(
                ctx.shared_ctx.python_version, ctx.shared_ctx.architectures
            ),
        )
        for distro_contexts in release_context.values()
        for ctx in distro_contexts
        for tag, meta in ctx.release_tags.items()
        for tr in target_releases
        if tr in tag
    }

    base_tags = dicttoolz.keyfilter(lambda x: "base" in x, hierarchy)
    build_tags = dicttoolz.keyfilter(lambda x: "base" not in x, hierarchy)

    # non "base" items
    if env.distros:
        base_, build_ = {}, {}
        for distro in env.distros:
            base_.update(dicttoolz.keyfilter(lambda x: distro in x, base_tags))
            build_.update(dicttoolz.keyfilter(lambda x: distro in x, build_tags))
        return base_, build_
    else:
        return base_tags, build_tags


def prepare_xx_image(ctx: Environment):
    builder = create_buildx_builder("prepare_xx_image")
    for arch in ctx.docker_target_arch:
        img = docker.image.list(filters={"reference": f"local-xx:*-{arch}"})
        if len(img) == 0:
            config = docker.buildx.bake(
                targets=f"local-xx-{arch}",
                builder=builder,
                load=True,
                progress="plain",
                files=ctx._fs.getsyspath("docker-bake.hcl"),
            )
            if ctx.verbose:
                logger.info(config)
