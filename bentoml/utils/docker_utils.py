import click
import sys
import os
import json
import re
import psutil
from bentoml.saved_bundle import load_bento_service_metadata
from bentoml.cli.utils import Spinner, echo_docker_api_result
from bentoml.exceptions import BentoMLException, CLIException
from bentoml.utils import resolve_bundle_path
from bentoml.cli.click_utils import (
    CLI_COLOR_WARNING,
    CLI_COLOR_SUCCESS,
    _echo,
    BentoMLCommandGroup,
    conditional_argument,
)


def to_valid_docker_image_name(name):
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return name.lower().strip("._-")


def to_valid_docker_image_version(version):
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return version.encode("ascii", errors="ignore").decode().lstrip(".-")[:128]


def validate_tag(ctx, param, tag):  # pylint: disable=unused-argument
    if tag is None:
        return tag

    if ":" in tag:
        name, version = tag.split(":")[:2]
    else:
        name, version = tag, None

    valid_name_pattern = re.compile(
        r"""
        ^(
        [a-z0-9]+      # alphanumeric
        (.|_{1,2}|-+)? # seperators
        )*$
        """,
        re.VERBOSE,
    )
    valid_version_pattern = re.compile(
        r"""
        ^
        [a-zA-Z0-9] # cant start with .-
        [ -~]{,127} # ascii match rest, cap at 128
        $
        """,
        re.VERBOSE,
    )

    if not valid_name_pattern.match(name):
        raise click.BadParameter(
            f"Provided Docker Image tag {tag} is invalid. "
            "Name components may contain lowercase letters, digits "
            "and separators. A separator is defined as a period, "
            "one or two underscores, or one or more dashes.",
            ctx=ctx,
            param=param,
        )
    if version and not valid_version_pattern.match(version):
        raise click.BadParameter(
            f"Provided Docker Image tag {tag} is invalid. "
            "A tag name must be valid ASCII and may contain "
            "lowercase and uppercase letters, digits, underscores, "
            "periods and dashes. A tag name may not start with a period "
            "or a dash and may contain a maximum of 128 characters.",
            ctx=ctx,
            param=param,
        )
    return tag


def containerize_bento_service(
    bento,
    push=False,
    tag=None,
    build_arg=None,
    username=None,
    password=None,
    pip_installed_bundle_path=None,
):
    """Containerize specified BentoService.

    BENTO is the target BentoService to be containerized, referenced by its name
    and version in format of name:version. For example: "iris_classifier:v1.2.0"

    `bentoml containerize` command also supports the use of the `latest` tag
    which will automatically use the last built version of your Bento.

    You can provide a tag for the image built by Bento using the
    `--docker-image-tag` flag. Additionally, you can provide a `--push` flag,
    which will push the built image to the Docker repository specified by the
    image tag.

    You can also prefixing the tag with a hostname for the repository you wish
    to push to.
    e.g. `bentoml containerize IrisClassifier:latest --push --tag username/iris`
    would build a Docker image called `username/iris:latest` and push that to
    Docker Hub.

    By default, the `containerize` command will use the credentials provided by
    Docker. You may provide your own through `--username` and `--password`.
    """
    saved_bundle_path = resolve_bundle_path(bento, pip_installed_bundle_path)
    _echo(f"bendle path is {pip_installed_bundle_path}")
    _echo(f"bento is {bento}")
    _echo(f"Found Bento: {saved_bundle_path}")

    bento_metadata = load_bento_service_metadata(saved_bundle_path)
    name = to_valid_docker_image_name(bento_metadata.name)
    version = to_valid_docker_image_version(bento_metadata.version)

    if not tag:
        _echo(
            "Tag not specified, using tag parsed from "
            f"BentoService: '{name}:{version}'"
        )
        tag = f"{name}:{version}"
    if ":" not in tag:
        _echo(
            "Image version not specified, using version parsed "
            f"from BentoService: '{version}'",
            CLI_COLOR_WARNING,
        )
        tag = f"{tag}:{version}"

    print("tag is ", tag)
    docker_build_args = {}
    if build_arg:
        for arg in build_arg:
            key, value = arg.split("=")
            docker_build_args[key] = value

    import docker

    docker_api = docker.APIClient()
    try:
        with Spinner(f"Building Docker image {tag} from {bento} \n"):
            for line in echo_docker_api_result(
                docker_api.build(
                    path=saved_bundle_path,
                    tag=tag,
                    decode=True,
                    buildargs=docker_build_args,
                )
            ):
                _echo(line)
    except docker.errors.APIError as error:
        raise CLIException(f"Could not build Docker image: {error}")

    _echo(
        f"Finished building {tag} from {bento}", CLI_COLOR_SUCCESS,
    )

    if push:
        auth_config_payload = (
            {"username": username, "password": password}
            if username or password
            else None
        )

        try:
            with Spinner(f"Pushing docker image to {tag}\n"):
                for line in echo_docker_api_result(
                    docker_api.push(
                        repository=tag,
                        stream=True,
                        decode=True,
                        auth_config=auth_config_payload,
                    )
                ):
                    _echo(line)
            _echo(
                f"Pushed {tag} to {name}", CLI_COLOR_SUCCESS,
            )
        except (docker.errors.APIError, BentoMLException) as error:
            raise CLIException(f"Could not push Docker image: {error}")
    return tag
