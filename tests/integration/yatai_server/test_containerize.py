import logging

from bentoml.yatai.client import get_yatai_client
from tests.bento_service_examples.example_bento_service import ExampleBentoService

logger = logging.getLogger('bentoml.test')


def test_yatai_server_containerize_without_push():
    svc = ExampleBentoService()
    svc.pack('model', [1, 2, 3])
    logger.info('Saving bento service to local yatai server')
    svc.save()

    yc = get_yatai_client()
    tag = 'mytag'
    built_tag = yc.repository.containerize(bento=f'{svc.name}:{svc.version}', tag=tag)
    assert built_tag == f'{tag}:{svc.version}'
