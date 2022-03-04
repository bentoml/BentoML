import atexit
import typing as t
import logging
from uuid import uuid1
from concurrent.futures import ThreadPoolExecutor

import click
from manager import SUPPORTED_OS_RELEASES
from manager import SUPPORTED_PYTHON_VERSION
from manager import DOCKERFILE_BUILD_HIERARCHY
from manager._utils import run
from manager._utils import as_posix
from manager._utils import graceful_exit
from manager._utils import get_docker_platform_mapping
from manager._utils import SUPPORTED_ARCHITECTURE_TYPE
from manager._context import ReleaseCtx
from manager._context import load_context
from manager._context import DOCKERFILE_NAME
from manager._container import ManagerContainer

DOCKER_BUILDX_BUILDER_NAME = "bentoml_{package}_multiarch_builder_{uuid}"

logger = logging.getLogger(__name__)


@graceful_exit
def add_build_command(cli: click.Group) -> None:
    @cli.command()
    @click.option(
        "--bentoml_version",
        required=True,
        type=click.STRING,
        help="targeted bentoml version",
    )
    @click.option(
        "--releases",
        required=False,
        type=click.Choice(DOCKERFILE_BUILD_HIERARCHY),
        multiple=True,
        help=f"Targeted releases for an image, default is to build all following the order: {DOCKERFILE_BUILD_HIERARCHY}",
        default=DOCKERFILE_BUILD_HIERARCHY,
    )
    @click.option(
        "--platforms",
        required=False,
        type=click.Choice(SUPPORTED_ARCHITECTURE_TYPE),
        multiple=True,
        help="Targeted a given platforms to build image on: linux/amd64,linux/arm64",
        default=SUPPORTED_ARCHITECTURE_TYPE,
    )
    @click.option(
        "--distros",
        required=False,
        type=click.Choice(SUPPORTED_OS_RELEASES),
        multiple=True,
        help="Targeted a distros releases",
        default=SUPPORTED_OS_RELEASES,
    )
    @click.option(
        "--python_version",
        required=False,
        type=click.Choice(SUPPORTED_PYTHON_VERSION),
        multiple=True,
        help=f"Targets a python version, default to {SUPPORTED_PYTHON_VERSION}",
        default=SUPPORTED_PYTHON_VERSION,
    )
    @click.option(
        "--max_workers",
        required=False,
        type=int,
        default=5,
        help="Defauls with # of workers used for ThreadPoolExecutor",
    )
    @click.option(
        "--generated_dir",
        type=click.STRING,
        metavar="generated",
        help=f"Output directory for generated Dockerfile, default to {ManagerContainer.generated_dir.as_posix()}",
        default=as_posix(ManagerContainer.generated_dir),
    )
    def build(
        docker_package: str,
        bentoml_version: str,
        generated_dir: str,
        releases: t.Tuple[str],
        platforms: t.Tuple[str],
        distros: t.Tuple[str],
        python_version: t.Tuple[str],
        max_workers: int,
    ) -> None:
        """
        Build releases docker images with `buildx`. Supports multithreading.

        Usage:
            manager build --bentoml_version 1.0.0a5
            manager build --bentoml_version 1.0.0a5 --releases base --max_workers 2
            manager build --bentoml_version 1.0.0a5 --docker_package runners

        \b
        By default we will generate all given specs defined under manifest/<docker_package>.yml
        """

        # We need to install QEMU to support multi-arch
        run(
            "docker",
            "run",
            "--rm",
            "--privileged",
            "tonistiigi/binfmt",
            "--install",
            "all",
            log_output=False,
        )

        builder_list = []

        build_ctx, release_ctx, _ = load_context(
            bentoml_version=bentoml_version,
            docker_package=docker_package,
            python_version=python_version,
            generated_dir=generated_dir,
        )
        base_tag, build_tag = _order_build_tags(
            release_ctx=release_ctx, releases=releases, distros=distros
        )

        def build_multi_arch(tags: t.Tuple[str, ...]) -> None:
            image_tag, docker_build_ctx, tag, *platforms = tags

            builder_name = DOCKER_BUILDX_BUILDER_NAME.format(
                package=docker_package, uuid=uuid1()
            )
            builder_list.append(builder_name)

            create_buildx_builder_instance(*platforms, name=builder_name)
            logger.info(
                f":brick: [bold yellow] Building {image_tag}...[/]",
                extra={"markup": True},
            )
            platform_string = ",".join(
                [v for k, v in get_docker_platform_mapping().items() if k in platforms]
            )
            python_version = build_ctx[tag].envars["PYTHON_VERSION"]

            run(
                "docker",
                "buildx",
                "build",
                "--push",
                "--platform",
                platform_string,
                "--progress",
                "plain",
                "--build-arg",
                f"PYTHON_VERSION={python_version}",
                "-t",
                image_tag,
                "--build-arg",
                "BUILDKIT_INLINE_CACHE=1",
                f"--cache-from=type=registry,ref={image_tag}",
                "-f",
                as_posix(docker_build_ctx, DOCKERFILE_NAME),
                ".",
                log_output=True,
            )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(build_multi_arch, base_tag))
                list(executor.map(build_multi_arch, build_tag))
        finally:
            for b in builder_list:
                run("docker", "buildx", "rm", b)
        atexit.register(run, "docker", "buildx", "prune")


def _order_build_tags(
    release_ctx: t.Dict[str, ReleaseCtx],
    releases: t.Tuple[str],
    distros: t.Tuple[str],
) -> t.Tuple[t.Tuple[str, ...], t.Tuple[str, ...]]:
    rct = [
        (k, x["output_path"], z, *v.shared_ctx.architectures)
        for z, v in release_ctx.items()
        for k, x in v.release_tags.items()
    ]

    base_tags = list(set((k, v, l, *a) for k, v, l, *a in rct if "base" in k))
    build_tags = []

    if releases and (releases in DOCKERFILE_BUILD_HIERARCHY):
        build_tags += [(k, v, l, *a) for k, v, l, *a in rct if releases in k]
    else:
        # non "base" items
        build_tags = []
        for release in DOCKERFILE_BUILD_HIERARCHY:
            build_tags += [
                (k, v, l, *a)
                for k, v, l, *a in rct
                if release != "base" and release in k
            ]
    build_tags = list(set((k, v, l, *a) for k, v, l, *a in build_tags))
    if distros is not None:
        base_tags = sorted(base_tags, key=lambda a: a[0] in distros)
        build_tags = sorted(build_tags, key=lambda a: a[0] in distros)

    return base_tags, build_tags


def create_buildx_builder_instance(*platform_args: str, name: str):
    # create one for each thread that build the given images

    platform_string = ",".join(
        [v for k, v in get_docker_platform_mapping().items() if k in platform_args]
    )
    run("docker", "buildx", "ls", log_output=False)
    run(
        "docker",
        "buildx",
        "create",
        "--use",
        "--platform",
        platform_string,
        "--driver-opt",
        "image=moby/buildkit:master",
        "--name",
        name,
        log_output=False,
    )
    run("docker", "buildx", "inspect", "--bootstrap", log_output=False)
