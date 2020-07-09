import contextlib

import docker

from .dependency_verification_service import gen_test_bundle


@contextlib.contextmanager
def docker_container():
    client = docker.from_env()
    test_dir = '../bento-tests/int_test'
    gen_test_bundle(test_dir)

    try:
        print('building img...')
        img, logs = client.images.build(path=test_dir, tag='dependency_test', rm=True)

    except docker.errors.BuildError:
        print(logs)

    print('running container...')
    container = client.containers.run('dependency_test', detach=True, remove=True)
    yield img

    print(f'Killing container {container.id}')
    container.kill()
    print(f'removing {img.id}..')
    client.images.remove(img.id)


def test_dependencies_of_docker_container():
    with docker_container() as img:
        print(img.id)
