import logging
import contextlib
import time

import docker
import requests

PORT = 50054
logger = logging.getLogger('bentoml.test')


@contextlib.contextmanager
def bento_docker_server(tag, path, port=PORT):
    dClient = docker.from_env()
    logger.info("Starting Build...")
    try:
        dClient.images.build(path=path, tag=tag)
    except docker.errors.BuildError as exec:
        # log build logs only if build fails
        for log in exec.build_log:
            print(log)
            logger.error(log)

    logger.info('Starting docker Server...')
    container = dClient.containers.run(
        name='bento-test-docker-init',
        image=tag,
        ports={5000: port},
        remove=True,
        detach=True,
    )
    yield container

    logger.info('Stopping docker Server...')
    container.kill()


def wait_till_server_up(timeout=30):
    """
    Keep trying till the gunicorn server is
    up and running. Takes a timeout parameter (in sec).
    """
    url = 'http://127.0.0.1:' + str(PORT) + '/healthz'
    connection_error = True
    total_time = 0
    while connection_error:
        try:
            requests.get(url)
            logger.info('server is up!')
            connection_error = False
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            total_time += 1
            if total_time >= timeout:
                raise TimeoutError('docker container is not yet up. Check logs')
