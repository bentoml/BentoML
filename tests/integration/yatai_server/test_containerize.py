import logging
import subprocess

from bentoml.yatai.client import get_yatai_client
from tests.bento_service_examples.example_bento_service import ExampleBentoService

logger = logging.getLogger('bentoml.test')


def test_yatai_server_containerize_without_push(example_bento_service_class):
    svc = example_bento_service_class()
    svc.pack('model', [1, 2, 3])
    logger.info('Saving bento service to local yatai server')
    svc.save()

    yc = get_yatai_client()
    tag = 'mytag'
    built_tag = yc.repository.containerize(bento=f'{svc.name}:{svc.version}', tag=tag)
    assert built_tag == f'{tag}:{svc.version}'


def test_yatai_server_containerize_from_cli(example_bento_service_class):
    svc = example_bento_service_class()
    svc.pack('model', [1, 2, 3])
    logger.info('Saving bento service to local yatai server')
    svc.save()
    bento_tag = f'{svc.name}:{svc.version}'
    tag = 'mytagfoo'

    command = [
        'bentoml',
        'containerize',
        bento_tag,
        '--build-arg',
        'EXTRA_PIP_INSTALL_ARGS=--extra-index-url=https://pypi.org',
        '-t',
        tag,
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout = docker_proc.stdout.read().decode('utf-8')
    assert f'{tag}:{svc.version}' in stdout, 'Failed to build container'
