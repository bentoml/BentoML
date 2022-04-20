import sys
import logging
import argparse

from manager.build import entrypoint as entrypoint_build
from manager.generate import entrypoint as entrypoint_generate
from manager._configuration import DockerManagerContainer

logger = logging.getLogger(__name__)


def cli() -> None:
    # fmt: off
    """

Manager: BentoML's Docker Images release management system.

Features:
    Multiple Python version: 3.7, 3.8, 3.9, 3.10
    Multiple platform: arm64v8, amd64, ppc64le
    Multiple Linux Distros that you love: Debian, Ubuntu, UBI, alpine

Get started with:
    $ manager --help
    """
    # fmt: on
    parser = argparse.ArgumentParser(prog="manager")
    parser.add_argument("--bentoml-version", type=str)
    parser.add_argument("--cuda-version", type=str)
    parser.add_argument("--organization", type=str)

    subparser = parser.add_subparsers()

    build_parser = subparser.add_parser(
        "build",
        help="build docker images from generated Dockerfiles.",
        parents=[parser],
        add_help=False,
    )
    build_parser.add_argument("--releases", action="append", nargs="+")
    build_parser.add_argument("--distros", action="append", nargs="+")
    build_parser.add_argument("--dry-run", action="store_true")
    build_parser.add_argument("--max-worker", type=int, default=5)
    build_parser.set_defaults(func=entrypoint_build)

    generate_parser = subparser.add_parser(
        "generate",
        help="generate Dockerfiles and README.md.",
        parents=[parser],
        add_help=False,
    )
    generate_parser.set_defaults(func=entrypoint_generate)

    args = parser.parse_args()

    if args.bentoml_version is not None:
        DockerManagerContainer.bentoml_version.set(args.bentoml_version)
    else:
        args.bentoml_version = DockerManagerContainer.bentoml_version.get()

    if args.cuda_version is not None:
        DockerManagerContainer.cuda_version.set(args.cuda_version)
    else:
        args.cuda_version = DockerManagerContainer.cuda_version.get()

    if args.organization is not None:
        DockerManagerContainer.organization.set(args.organization)
    else:
        args.organization = DockerManagerContainer.organization.get()

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted. Exitting...")
        sys.exit(0)
    except Exception as e:
        raise e


if __name__ == "__main__":
    cli()
