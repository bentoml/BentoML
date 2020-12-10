import logging

from bentoml.yatai.client import get_yatai_client
from tests.bento_service_examples.example_bento_service import ExampleBentoService
from tests.yatai.local_yatai_service import local_yatai_service_from_cli

logger = logging.getLogger('bentoml.test')


def test_yatai_server_containerize_without_push():
    with local_yatai_service_from_cli() as yatai_server_url:
        svc = ExampleBentoService()
        svc.pack('model', [1, 2, 3])
        logger.info('Saving bento service to local yatai server')
        svc.save(yatai_url=yatai_server_url)

        yc = get_yatai_client(yatai_server_url)
        tag = 'mytag'
        result = yc.repository.containerize(
            bento=f'{svc.name}:{svc.version}', tag=tag
        )
        assert result.tag == f'{tag}:{svc.version}'