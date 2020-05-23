import subprocess
import contextlib
import logging
import os
import uuid
import time

import docker

from bentoml.deployment.utils import ensure_docker_available_or_raise

logger = logging.getLogger('bentoml.test')


def wait_for_docker_container_ready(container_name, check_message):
    docker_client = docker.from_env()

    start_time = time.time()
    while True:
        time.sleep(1)
        container_list = docker_client.containers.list(
            filters={'name': container_name, 'status': 'running'}
        )
        logger.info(f'Waiting for container {container_name}')
        logger.info("Container list: " + str(container_list))
        if not container_list:
            # Raise timeout, if take more than 60 seconds
            if time.time() - start_time > 60:
                raise TimeoutError(f'Get container: {container_name} times out')
            else:
                continue

        assert len(container_list) == 1, 'should be only one container running'

        yatai_service_container = container_list[0]

        logger.info('CONTAINER_LOG: ' + yatai_service_container.logs().decode('utf-8'))
        if check_message in yatai_service_container.logs():
            break


def cleanup_docker_containers():
    docker_client = docker.from_env()
    container_list = docker_client.containers.list(filters={'name': 'bentoml-e2e-test'})
    logger.info('Delete previous container')
    for container in container_list:
        logger.debug(f'Killing container {container.name}')
        container.kill()


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


@contextlib.contextmanager
def start_yatai_server(
    docker_image,
    db_url=None,
    db_host_name=None,
    s3_endpoint_url=None,
    repo_base_url='/tmp',
    grpc_port=50051,
    ui_port=3000,
    env=None,
):
    ensure_docker_available_or_raise()
    container_name = f'e2e-test-yatai-service-container-{uuid.uuid4().hex[:6]}'
    yatai_service_url = f'localhost:{grpc_port}'

    command = [
        'docker',
        'run',
        '--rm',
        '--name',
        container_name,
        '-p',
        f'{grpc_port}:50051',
        '-p',
        f'{ui_port}:3000',
    ]
    if env:
        for key in env:
            command.extend(['-e', f'{key}={env[key]}'])
    if db_host_name:
        command.extend(['--link', f'{db_host_name}:postgres-container'])
    command.extend([docker_image, '--repo-base-url', repo_base_url])
    if db_url:
        command.extend(['--db-url', db_url])
    if s3_endpoint_url:
        command.extend(['--s3-endpoint-url', s3_endpoint_url])

    logger.info(f'Running docker command {" ".join(command)}')

    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_for_docker_container_ready(
        container_name, b'* Starting BentoML YataiService gRPC Server'
    )
    yield yatai_service_url
    docker_proc.terminate()


@contextlib.contextmanager
def start_postgres_docker():
    ensure_docker_available_or_raise()
    cleanup_docker_containers()
    container_name = (
        f'bentoml-e2e-test-yatai-service-postgres-db-{uuid.uuid4().hex[:6]}'
    )
    db_url = f'postgresql://postgres:postgres@postgres-container:5432/bentoml'

    command = [
        'docker',
        'run',
        '--rm',
        '--name',
        container_name,
        '-e',
        'POSTGRES_PASSWORD=postgres',
        '-p',
        '5432:5432',
        'postgres',
    ]
    logger.debug(f'Docker command: {" ".join(command)}')
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_for_docker_container_ready(
        container_name, b'database system is ready to accept connections'
    )

    from sqlalchemy_utils import create_database

    create_database(
        f'postgresql://postgres:postgres@localhost:5432/bentoml'
    )
    yield {'url': db_url, 'container_name': container_name}
    docker_proc.terminate()
