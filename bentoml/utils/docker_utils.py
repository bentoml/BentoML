import re
import logging
from bentoml.exceptions import YataiDeploymentException

logger = logging.getLogger(__name__)


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
        raise YataiDeploymentException(
            f"Provided Docker Image tag {tag} is invalid. "
            "Name components may contain lowercase letters, digits "
            "and separators. A separator is defined as a period, "
            "one or two underscores, or one or more dashes."
        )
    if version and not valid_version_pattern.match(version):
        raise YataiDeploymentException(
            f"Provided Docker Image tag {tag} is invalid. "
            "A tag name must be valid ASCII and may contain "
            "lowercase and uppercase letters, digits, underscores, "
            "periods and dashes. A tag name may not start with a period "
            "or a dash and may contain a maximum of 128 characters."
        )
    return tag


def containerize_bento_service(
    bento_name,
    bento_version,
    saved_bundle_path,
    push=False,
    tag=None,
    build_arg=None,
    username=None,
    password=None,
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
    name = to_valid_docker_image_name(bento_name)
    version = to_valid_docker_image_version(bento_version)

    if not tag:
        tag = f"{name}:{version}"
    if ":" not in tag:
        tag = f"{tag}:{version}"

    docker_build_args = {}
    if build_arg:
        for arg in build_arg:
            key, value = arg.split("=")
            docker_build_args[key] = value

    import docker

    docker_api = docker.APIClient()
    try:
        logger.info("Building image")
        for line in docker_api.build(
            path=saved_bundle_path, tag=tag, decode=True, buildargs=docker_build_args,
        ):
            logger.debug(line)
    except docker.errors.APIError as error:
        raise YataiDeploymentException(f"Could not build Docker image: {error}")

    if push:
        auth_config_payload = (
            {"username": username, "password": password}
            if username or password
            else None
        )

        try:
            logger.info("Pushing image")
            for line in docker_api.push(
                repository=tag,
                stream=True,
                decode=True,
                auth_config=auth_config_payload,
            ):
                logger.debug(line)
        except docker.errors.APIError as error:
            raise YataiDeploymentException(f"Could not push Docker image: {error}")
    return tag
