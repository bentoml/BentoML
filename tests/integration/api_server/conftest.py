# pylint: disable=redefined-outer-name
import time
import pytest
import bentoml

from .example_service import gen_test_bundle


@pytest.fixture(params=[True, False], scope="session")
def enable_microbatch(request):
    pytest.enable_microbatch = request.param
    return request.param


@pytest.fixture(autouse=True, scope='session')
def image(tmpdir_factory):
    import docker

    client = docker.from_env()

    bundle_dir = tmpdir_factory.mktemp('test_bundle')
    bundle_path = str(bundle_dir)
    gen_test_bundle(bundle_path)
    image = client.images.build(path=bundle_path, tag="example_service", rm=True)[0]
    yield image
    client.images.remove(image.id)


@pytest.fixture(autouse=True, scope='session')
def host(image, enable_microbatch):
    import docker

    client = docker.from_env()

    with bentoml.utils.reserve_free_port() as port:
        pass
    if enable_microbatch:
        command = "bentoml serve-gunicorn /bento --enable-microbatch --workers 1"
    else:
        command = "bentoml serve-gunicorn /bento --workers 1"
    container = client.containers.run(
        command=command,
        image=image.id,
        auto_remove=True,
        tty=True,
        ports={'5000/tcp': port},
        detach=True,
    )
    time.sleep(10)
    yield f"127.0.0.1:{port}"
    container.stop()
    time.sleep(1)
