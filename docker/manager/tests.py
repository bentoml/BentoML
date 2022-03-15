import typing as t
import logging
import traceback
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import fs
import click

from ._internal.utils import run
from ._internal.utils import DOCKERFILE_BUILD_HIERARCHY
from ._internal.utils import SUPPORTED_ARCHITECTURE_TYPE
from ._internal.groups import Environment
from ._internal.groups import pass_environment
from ._internal.exceptions import ManagerException

logger = logging.getLogger(__name__)


def add_tests_command(cli: click.Group) -> None:
    @cli.command(name="run-tests")
    @click.option(
        "--releases",
        required=False,
        type=click.Choice(DOCKERFILE_BUILD_HIERARCHY),
        multiple=True,
        help=f"Targets releases for an image, default is to build all following the order: {DOCKERFILE_BUILD_HIERARCHY}",
        default=DOCKERFILE_BUILD_HIERARCHY,
    )
    @click.option(
        "--platforms",
        required=False,
        type=click.Choice(SUPPORTED_ARCHITECTURE_TYPE),
        multiple=True,
        help="Targets a given platforms to build image on: linux/amd64,linux/arm64",
        default=SUPPORTED_ARCHITECTURE_TYPE,
    )
    @click.option(
        "--max-workers",
        required=False,
        type=int,
        default=5,
        help="Defauls with # of workers used for ThreadPoolExecutor",
    )
    @pass_environment
    def run_tests(
        ctx: Environment,
        releases: t.Tuple[str],
        platforms: t.Tuple[str],
        max_workers: int,
    ) -> None:
        """
        Run tests per releases.

        \b
        For now we will do sanity check for runtime and cudnn releases.
        Reasons being there is no need to tests devel for now (since it will just install BentoML from git@main)
        """
        # sanity check
        logger.warning("This command will run with sudo. Use with care!")
        test_shell_scripts = fs.path.combine("tests", "sanity_check.sh")

        def run_sanity_check_tests(args: t.Tuple[str, str, str, str, str]):
            bentoml_version, releases, platform, distro, python_version = args
            run(
                "sudo",
                "bash",
                f"{ctx._fs.getsyspath(test_shell_scripts)}",
                "--image_name",
                ctx.docker_package,
                "--bentoml_version",
                bentoml_version,
                "--python_version",
                python_version,
                "--distros",
                distro,
                "--suffix",
                releases,
                "--platforms",
                platform,
            )

        mapping = product(
            [ctx.bentoml_version],
            [r for r in releases if r not in ["base", "devel"]],
            platforms,
            ctx.distros,
            ctx.python_version,
        )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(run_sanity_check_tests, mapping))
        except Exception as e:  # pylint: disable
            traceback.print_exc()
            raise ManagerException("Error while running tests.") from e
