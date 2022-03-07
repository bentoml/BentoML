import os
import atexit
import typing as t
import logging
from uuid import uuid1
from concurrent.futures import ThreadPoolExecutor

import click
from manager._utils import run
from manager._utils import as_posix
from manager._utils import graceful_exit
from manager._utils import DOCKERFILE_NAME
from manager._utils import SUPPORTED_REGISTRIES
from manager._utils import SUPPORTED_PYTHON_VERSION
from manager._utils import DOCKERFILE_BUILD_HIERARCHY
from manager._utils import get_docker_platform_mapping
from manager._schemas import ReleaseCtx
from manager._exceptions import ManagerBuildFailed
from manager._click_utils import Environment
from manager._click_utils import pass_environment

logger = logging.getLogger(__name__)

BUILDER_LIST = []

REGISTRIES_ENVARS_MAPPING = {"docker.io": "DOCKER_URL", "ecr": "AWS_URL"}


@graceful_exit
def add_build_command(cli: click.Group) -> None:
    @cli.command()
    @click.option(
        "--releases",
        required=False,
        type=click.Choice(DOCKERFILE_BUILD_HIERARCHY),
        multiple=True,
        help=f"Targets releases for an image, default is to build all following the order: {DOCKERFILE_BUILD_HIERARCHY}",
        default=DOCKERFILE_BUILD_HIERARCHY,
    )
    @click.option(
        "--registry",
        required=False,
        type=click.Choice(SUPPORTED_REGISTRIES),
        default=SUPPORTED_REGISTRIES,
        help="Targets registry to login.",
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
        ctx: Environment, releases: t.Tuple[str], max_workers: int, registry: str
    ) -> None:
        """
        Build releases docker images with `buildx`. Supports multithreading.

        \b
        Usage:
            manager build --bentoml-version 1.0.0a5 --registry ecr
            manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2
            manager build --bentoml-version 1.0.0a5 --releases base --max-workers 2 --python-version 3.8 --python-version 3.9
            manager build --bentoml-version 1.0.0a5 --docker-package <other-package>

        \b
        By default we will generate all given specs defined under manifest/<docker_package>.yml
        """
        # We need to install QEMU to support multi-arch
        logger.warning(
            "Make sure to run [bold red]make emulator[/] to install all required QEMU.",
            extra={"markup": True},
        )
        global BUILDER_LIST

        base_tag, build_tag = order_build_hierarchy(
            release_ctx=ctx.release_ctx, releases=releases, distros=ctx.distros
        )
        cmd_base_tag = [
            (platform_string, run_args)
            for tags in base_tag
            for platform_string, run_args in docker_buildx_cmd(tags, ctx, registry)
        ]
        cmd_build_tag = [
            (platform_string, run_args)
            for tags in build_tag
            for platform_string, run_args in docker_buildx_cmd(tags, ctx, registry)
        ]

        def build_multi_arch(args: t.Tuple[str, t.List[str]]) -> None:
            platform_string, run_args = args

            builder_name = f"bentoml_{ctx.docker_package}_multiarch_builder_{uuid1()}"

            BUILDER_LIST.append(builder_name)
            create_buildx_builder_instance(
                platform_string, name=builder_name, verbose_=ctx.verbose
            )

            try:
                run(run_args[0], *run_args[1:], log_output=True)
            except Exception:
                run("docker", "buildx", "rm", builder_name)

        try:
            with ThreadPoolExecutor(
                max_workers=max_workers * len(SUPPORTED_PYTHON_VERSION)
            ) as executor:
                list(executor.map(build_multi_arch, cmd_base_tag))
                list(executor.map(build_multi_arch, cmd_build_tag))
            executor.shutdown()
            atexit.register(run, "docker", "buildx", "prune")
        finally:
            # tries to remove zombie proc.
            for b in BUILDER_LIST:
                run("docker", "buildx", "rm", b)


def docker_buildx_cmd(
    tags: t.Tuple[str, str, str, t.Tuple[str, ...]], ctx: Environment, registry: str
) -> t.Generator[t.Tuple[str, t.List[str]], None, None]:

    image_tag, docker_build_ctx, distro, *platforms = tags

    python_version = ctx.build_ctx[distro][0].envars["PYTHON_VERSION"]
    tag_prefix = os.environ.get(REGISTRIES_ENVARS_MAPPING[registry], None)
    if tag_prefix is None:
        raise ManagerBuildFailed("Failed to retrieve URL prefix for docker registry.")

    baseline = [
        "docker",
        "buildx",
        "build",
    ]
    general_kwargs = [
        "--progress",
        "plain",
        "--build-arg",
        f"PYTHON_VERSION={python_version}",
        "--build-arg",
        "BUILDKIT_INLINE_CACHE=1",
        f"--cache-from=type=registry,ref={image_tag}",
        "-f",
        as_posix(docker_build_ctx, DOCKERFILE_NAME),
    ]

    if "base" in image_tag:
        # We don't want to release base images
        # We need to load this image back into docker memory in order to use of for higher hierarchy build
        platforms_list = [
            ["--load", "--platform", v]
            for k, v in get_docker_platform_mapping().items()
            if k in platforms
        ]
        for k in platforms_list:
            yield (k[2], baseline + k + general_kwargs + ["--tag", image_tag] + ["."])
    else:
        # Now each release images would have tag_prefix append.
        # We can then use buildx build for multiple platform here.
        platform_string = ",".join(
            [v for k, v in get_docker_platform_mapping().items() if k in platforms]
        )
        yield (
            platform_string,
            baseline
            + ["--push", "--platform", platform_string]
            + general_kwargs
            + ["--tag", f"{tag_prefix}/{image_tag}"]
            + ["."],
        )


def order_build_hierarchy(
    release_ctx: t.Dict[str, t.List[ReleaseCtx]],
    releases: t.Optional[t.Tuple[str]],
    distros: t.Optional[t.List[str]],
) -> t.Tuple[t.List[t.Any], t.List[t.Any]]:
    """
    Returns [(image_tag, docker_build_context, distro, *platforms), ...] for base and other tags
    """
    orders = {v: k for k, v in enumerate(DOCKERFILE_BUILD_HIERARCHY)}
    if releases and all(i in DOCKERFILE_BUILD_HIERARCHY for i in releases):
        target_releases = tuple(sorted(releases, key=lambda k: orders[k]))
    else:
        target_releases = DOCKERFILE_BUILD_HIERARCHY

    if "base" not in target_releases:
        target_releases = ("base",) + target_releases

    rct = [
        (tag, meta["output_path"], distro, *ctx.shared_ctx.architectures)
        for distro, l_rltx in release_ctx.items()
        for ctx in l_rltx
        for tr in target_releases
        for tag, meta in ctx.release_tags.items()
        if tr in tag
    ]

    base_tags = [(k, v, l, *a) for k, v, l, *a in rct if "base" in k]
    build_tags = []

    # non "base" items
    for release in target_releases:
        build_tags += [
            (k, v, l, *a) for k, v, l, *a in rct if release != "base" and release in k
        ]
    if distros:
        base_tags = [(k, v, l, *a) for d in distros for k, v, l, *a in rct if d in k]
        build_tags = [(k, v, l, *a) for d in distros for k, v, l, *a in rct if d in k]

    return base_tags, build_tags


def create_buildx_builder_instance(
    platform_string: str, *, name: str, verbose_: bool = False
):
    # create one for each thread that build the given images

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
    if verbose_:
        run("docker", "buildx", "inspect", "--bootstrap", log_output=False)
