# type: ignore[reportUnusedFunction]
import sys
import typing as t

import click

from bentoml.bentos import containerize as containerize_bento

from ..utils.docker import validate_tag


def add_containerize_command(cli: click.Group) -> None:
    @cli.command(
        help="Containerizes given Bento into a ready-to-use Docker image.",
    )
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-t",
        "--docker-image-tag",
        help="Docker image tag, default to same as the Bento tag",
        required=False,
        callback=validate_tag,
    )
    @click.option("--build-arg", multiple=True, help="docker image build args")
    @click.option("--label", multiple=True, help="docker image label")
    @click.option("--no-cache", is_flag=True, default=False)
    @click.option("--platform", default=None)
    def containerize(
        bento_tag: str,
        docker_image_tag: str,
        build_arg: t.List[str],
        label: t.List[str],
        no_cache: bool,
        platform: str,
    ) -> None:
        """Containerize specified Bento.

        BENTO is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: "iris_classifier:v1.2.0"

        `bentoml containerize` command also supports the use of the `latest` tag
        which will automatically use the last built version of your Bento.

        You can provide a tag for the image built by Bento using the
        `--tag` flag. Additionally, you can provide a `--push` flag,
        which will push the built image to the Docker repository specified by the
        image tag.

        You can also prefixing the tag with a hostname for the repository you wish
        to push to.
        e.g. `bentoml containerize IrisClassifier:latest --push --tag
        repo-address.com:username/iris` would build a Docker image called
        `username/iris:latest` and push that to docker repository at repo-address.com.

        By default, the `containerize` command will use the current credentials
        provided by Docker daemon.
        """
        labels = {}
        if label:
            for label_str in label:
                key, value = label_str.split("=")
                labels[key] = value
        build_args = {}
        if build_arg:
            for build_arg_str in build_arg:
                key, value = build_arg_str.split("=")
                build_args[key] = value

        exit_code = not containerize_bento(
            bento_tag,
            docker_image_tag=docker_image_tag,
            build_args=build_args,
            labels=labels,
            no_cache=no_cache,
            platform=platform,
        )
        sys.exit(exit_code)
