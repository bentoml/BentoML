# pylint: disable=redefined-outer-name
import time
import urllib

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


def _wait_until_ready(_host, timeout, check_interval=0.5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if (
                urllib.request.urlopen(f'http://{_host}/healthz', timeout=0.1).status
                == 200
            ):
                break
        except Exception:  # pylint:disable=broad-except
            time.sleep(check_interval - 0.1)
    else:
        raise AssertionError(f"server didn't get ready in {timeout} seconds")


@pytest.fixture(autouse=True)
def host(image, enable_microbatch):
    import docker

    client = docker.from_env()

    with bentoml.utils.reserve_free_port() as port:
        pass
    if enable_microbatch:
        command = "bentoml serve-gunicorn /bento --enable-microbatch --workers 1"
    else:
        command = "bentoml serve-gunicorn /bento --workers 1"
    try:
        container = client.containers.run(
            command=command,
            image=image.id,
            auto_remove=True,
            tty=True,
            ports={'5000/tcp': port},
            detach=True,
        )
        _host = f"127.0.0.1:{port}"
        _wait_until_ready(_host, 10)
        yield _host
    finally:
        container.stop()
        time.sleep(1)  # make sure container stopped & deleted
