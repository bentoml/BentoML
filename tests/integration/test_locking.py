import logging
import pytest
import subprocess

from tests.bento_service_examples.example_bento_service import ExampleBentoService
from tests.threading_util import ThreadWithResult, run_delayed_thread

logger = logging.getLogger('bentoml.test')


def cli(svc, cmd, *args):
    bento_tag = f'{svc.name}:{svc.version}'
    proc = subprocess.Popen(
        ['bentoml', cmd, bento_tag, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.read().decode('utf-8')


@pytest.fixture()
def packed_svc():
    svc = ExampleBentoService()
    svc.pack('model', [1, 2, 3])
    svc.save()
    return svc


def test_write_lock_on_read_lock(packed_svc):
    containerize_thread = ThreadWithResult(
        target=cli, args=(packed_svc, 'containerize', '-t', 'imagetag')
    )
    delete_thread = ThreadWithResult(target=cli, args=(packed_svc, 'delete', '-y'))
    run_delayed_thread(containerize_thread, delete_thread)

    assert (
        f'Build container image: imagetag:{packed_svc.version}'
        in containerize_thread.result
    )
    assert (
        "Failed to acquire write lock, another lock held. Retrying"
        in delete_thread.result
    )
    assert f"Deleted {packed_svc.name}:{packed_svc.version}" in delete_thread.result


def test_read_lock_on_read_lock(packed_svc):
    containerize_thread = ThreadWithResult(
        target=cli, args=(packed_svc, 'containerize', '-t', 'imagetag')
    )
    get_thread = ThreadWithResult(target=cli, args=(packed_svc, 'get'))
    run_delayed_thread(containerize_thread, get_thread)

    assert (
        f'Build container image: imagetag:{packed_svc.version}'
        in containerize_thread.result
    )
    assert f'"name": "{packed_svc.name}"' in get_thread.result
    assert f'"version": "{packed_svc.version}"' in get_thread.result
