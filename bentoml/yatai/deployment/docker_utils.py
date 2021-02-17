import logging
import json

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
