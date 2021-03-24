import logging
import time

from bentoml.yatai.client import get_yatai_client
from bentoml.yatai.proto.repository_pb2 import BentoUri
from tests.bento_service_examples.example_bento_service import ExampleBentoService
from tests.yatai.local_yatai_service import (
    local_yatai_service_from_cli,
)

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_postgres_and_local_storage():
    postgres_db_url = 'postgresql://postgres:postgres@localhost/bentoml:5432'

    from sqlalchemy_utils import create_database

    create_database(postgres_db_url)
    time.sleep(60)

    with local_yatai_service_from_cli(db_url=postgres_db_url) as yatai_server_url:
        logger.info('Saving bento service')
        logger.info(f'yatai url is {yatai_server_url}')
        svc = ExampleBentoService()
        svc.pack('model', [1, 2, 3])
        bento_tag = f'{svc.name}:{svc.version}'
        logger.info(f'Saving BentoML saved bundle {bento_tag}')
        svc.save(yatai_url=yatai_server_url)

        yc = get_yatai_client(yatai_server_url)
        bento_pb = yc.repository.get(bento_tag)
        assert (
                bento_pb.uri.type == BentoUri.LOCAL
        ), 'BentoService storage type mismatched, expect LOCAL'
