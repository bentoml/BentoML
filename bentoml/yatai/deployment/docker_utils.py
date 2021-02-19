import logging
import json
from urllib.parse import urlparse

import docker

from bentoml.exceptions import MissingDependencyException, BentoMLException


logger = logging.getLogger(__name__)


def ensure_docker_available_or_raise():
    try:
        client = docker.from_env()
        client.ping()
    except docker.errors.APIError as error:
        raise MissingDependencyException(f'Docker server is not responsive. {error}')
    except docker.errors.DockerException:
        raise MissingDependencyException(
            'Docker is required for this deployment. Please visit '
            'www.docker.com for instructions'
        )


def process_docker_api_line(payload):
    """ Process the output from API stream, throw an Exception if there is an error """
    # Sometimes Docker sends to "{}\n" blocks together...
    errors = []
    for segment in payload.decode("utf-8").strip().split("\n"):
        line = segment.strip()
        if line:
            try:
                line_payload = json.loads(line)
            except ValueError as e:
                logger.warning("Could not decipher payload from Docker API: %s", str(e))
            if line_payload:
                if "errorDetail" in line_payload:
                    error = line_payload["errorDetail"]
                    error_msg = 'Error running docker command: {}: {}'.format(
                        error["code"], error['message']
                    )
                    logger.error(error_msg)
                    errors.append(error_msg)
                elif "stream" in line_payload:
                    logger.info(line_payload['stream'])

    if errors:
        error_msg = ";".join(errors)
        raise BentoMLException("Error running docker command: {}".format(error_msg))


def _strip_scheme(url):
    """ Stripe url's schema
    e.g.   http://some.url/path -> some.url/path
    :param url: String
    :return: String
    """
    parsed = urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, "", 1)


def generate_docker_image_tag(image_name, version='latest', registry_url=None):
    image_tag = f'{image_name}:{version}'.lower()
    if registry_url is not None:
        return _strip_scheme(f'{registry_url}/{image_tag}')
    else:
        return image_tag


def build_docker_image(context_path, dockerfile, image_tag, additional_build_args=None):
    docker_client = docker.from_env()
    try:
        docker_client.images.build(
            path=context_path,
            tag=image_tag,
            dockerfile=dockerfile,
            buildargs=additional_build_args,
        )
    except (docker.errors.APIError, docker.errors.BuildError) as error:
        logger.error(f'Failed to build docker image {image_tag}: {error}')
        raise BentoMLException(f'Failed to build docker image {image_tag}: {error}')


def push_docker_image_to_repository(
    repository, image_tag=None, username=None, password=None
):
    docker_client = docker.from_env()
    docker_push_kwags = {'repository': repository, 'tag': image_tag}
    if username is not None and password is not None:
        docker_push_kwags['auth_config'] = {'username': username, 'password': password}
    try:
        docker_client.images.push(**docker_push_kwags)
    except docker.errors.APIError as error:
        raise BentoMLException(f'Failed to push docker image {image_tag}: {error}')
