import logging

from bentoml.yatai.client import get_yatai_client
from bentoml.yatai.proto.repository_pb2 import BentoUri
from tests.bento_service_examples.example_bento_service import ExampleBentoService
from tests.yatai.local_yatai_service import local_yatai_server

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_postgres_and_local_storage():
    postgres_db_url = 'postgresql://postgres:postgres@localhost:5432'

    with local_yatai_server(db_url=postgres_db_url) as yatai_server_url:
        logger.info('Saving bento service')
        yc = get_yatai_client(yatai_server_url)
        svc = ExampleBentoService()
        svc.pack('model', [1, 2, 3])
        bento_tag = f'{svc.name}:{svc.version}'
        logger.info(f'Saving BentoML saved bundle {bento_tag}')
        svc.save(yatai_url=yatai_server_url)

        bento_pb = yc.repository.get(bento_tag)
        assert (
            bento_pb.uri.type == BentoUri.LOCAL
        ), 'BentoService storage type mismatched, expect LOCAL'

        logger.info(f'Deleting saved bundle {bento_tag}')
        delete_svc_result = yc.repository.delete(bento_tag)
        assert delete_svc_result is None
