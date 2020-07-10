import contextlib
import json
import time

import docker
import requests

from .dependency_verification_service import gen_test_bundle


@contextlib.contextmanager
def docker_container(port=5000):
    client = docker.from_env()
    # TODO: make test_dest dir according to travis fs
    test_dir = '../bento-tests/int_test'
    gen_test_bundle(test_dir)

    try:
        print('building img...')
        img, logs = client.images.build(path=test_dir, tag='dependency_test', rm=True)

    except docker.errors.BuildError:
        print(logs)

    print('running container...')
    container = client.containers.run('dependency_test', detach=True, remove=True,
                                      ports={5000: port})

    yield

    print(f'Killing container {container.id}')
    container.kill()

def wait_till_container_is_up(url, timeout=10):
    running_time=0
    while(running_time<timeout):
        try:
            r = requests.post(url, json='')
            print(r.status_code, r.text)
            if r.status_code == 200:
                return

        except requests.exceptions.ConnectionError:
            # the container is still booting up
            time.sleep(1)
        running_time += 1

def test_dependencies_of_docker_container():
    # TODO: change it to 5005
    port = 5000
    with docker_container(port=port):
        url = f'http://0.0.0.0:{port}/test_packages'
        wait_till_container_is_up(url)
        r = requests.post(url, json='')
        print(r.json())
        flags = json.loads(r.json())

        # check the returned flags to see if all the dependencies where installed
        for lib in flags:
            print(f'library: {lib} is installed in container: {flags[lib]}')
            assert flags[lib] is True
