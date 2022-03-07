import typing as t
import logging
import traceback
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import click
from manager import SUPPORTED_OS_RELEASES
from manager import SUPPORTED_PYTHON_VERSION
from manager import DOCKERFILE_BUILD_HIERARCHY
from manager._utils import run
from manager._utils import as_posix
from manager._utils import raise_exception
from manager._utils import SUPPORTED_ARCHITECTURE_TYPE
from manager._container import ManagerContainer
from manager.exceptions import ManagerException

logger = logging.getLogger(__name__)


def add_tests_command(cli: click.Group) -> None:
    @cli.command(name="run-tests")
    @click.option(
        "--bentoml-version",
        required=True,
        type=click.STRING,
        metavar="<x.y.z{a}>",
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
        "--python-version",
        required=False,
        type=click.Choice(SUPPORTED_PYTHON_VERSION),
        multiple=True,
        help=f"Targets a python version, default to {SUPPORTED_PYTHON_VERSION}",
        default=SUPPORTED_PYTHON_VERSION,
    )
    @click.option(
        "--max-workers",
        required=False,
        type=int,
        default=5,
        help="Defauls with # of workers used for ThreadPoolExecutor",
    )
    @raise_exception
    def run_tests(
        docker_package: str,
        bentoml_version: str,
        releases: t.Tuple[str],
        platforms: t.Tuple[str],
        distros: t.Tuple[str],
        python_version: t.Tuple[str],
        max_workers: int,
        generated_dir: str = ManagerContainer.generated_dir.as_posix(),
    ) -> None:
        """
        Run tests per releases.

        \b
        For now we will do sanity check for runtime and cudnn releases.
        Reasons being there is no need to tests devel for now (since it will just install BentoML from git@main)
        """
        # sanity check
        logger.warning("This command will run with sudo. Use with care!")
        test_shell_scripts = as_posix(
            ManagerContainer.tests_dir.joinpath("sanity_check.sh")
        )

        def run_sanity_check_tests(args: t.Tuple[str, str, str, str, str]):
            bentoml_version, releases, platform, distro, python_version = args
            run(
                "sudo",
                "bash",
                f"{test_shell_scripts}",
                "--image_name",
                docker_package,
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
            [bentoml_version],
            [r for r in releases if r not in ["base", "devel"]],
            platforms,
            distros,
            python_version,
        )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(run_sanity_check_tests, mapping))
        except Exception as e:  # pylint: disable
            traceback.print_exc()
            raise ManagerException("Error while running tests.") from e
