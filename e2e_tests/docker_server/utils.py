import os
import logging
import contextlib
import time
import json

import docker
import requests

PORT = 50054
logger = logging.getLogger('bentoml.test')


@contextlib.contextmanager
def bento_docker_server(tag, path, port=PORT):
    logger.info("Starting Build...")
    dClient = docker.APIClient(base_url='unix://var/run/docker.sock')
    logger.info("Docker client connected!")
    logger.info(dClient.version())
    logger.info(os.listdir(path))
    logger.info(dClient.containers())

    generator = dClient.build(path=path, tag=tag, rm=False)

    # output build logs
    while True:
        try:
            output = generator.__next__()
            output = output.strip(b'\r\n')
            json_output = json.loads(output)
            if 'stream' in json_output:
                logger.info(json_output['stream'].strip('\n'))
        except StopIteration:
            logger.info("Docker image build complete.")
            break
        except ValueError:
            logger.error("Error parsing output from docker image build: %s" % output)

    logger.info('Starting docker Server...')
    logger.info(dClient.containers())
    logger.info(dClient.images(all=True))
    container = dClient.create_container(
        name='bento-test-docker-init',
        image=tag,
        ports=[5000],
        host_config=dClient.create_host_config(port_bindings={
            5000:PORT
        })
    )

    dClient.start(container=container.get('Id'))

    yield

    logger.info('Stopping docker Server...')
    dClient.kill(container=container.get('Id'))


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
