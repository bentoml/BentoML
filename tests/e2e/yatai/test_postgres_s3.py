import logging

from bentoml.yatai.proto.repository_pb2 import BentoUri
from bentoml.yatai.client import get_yatai_client
from tests.bento_service_examples.example_bento_service import ExampleBentoService
from tests.yatai.local_yatai_service import local_yatai_service_container

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_sqlite_and_s3():
    # Note: Use pre-existing bucket instead of newly created bucket, because the
    # bucket's global DNS needs time to get set up.
    # https://github.com/boto/boto3/issues/1982#issuecomment-511947643
    s3_bucket_name = 's3://bentoml-e2e-test-repo/'

    with local_yatai_service_container(
        repo_base_url=s3_bucket_name
    ) as yatai_service_url:
        yc = get_yatai_client(yatai_service_url)
        logger.info('Saving bento service')
        svc = ExampleBentoService()
        svc.pack('model', [1, 2, 3])
        svc.save(yatai_url=yatai_service_url)
        bento_tag = f'{svc.name}:{svc.version}'
        logger.info('BentoService saved')

        logger.info("Display bento service info")
        bento = yc.repository.get(bento_tag)
        logger.info(bento)
        assert (
            bento.uri.type == BentoUri.S3
        ), 'BentoService storage type mismatched, expect S3'

        logger.info(f'Deleting saved bundle {bento_tag}')
        yc.repository.delete(bento_tag=bento_tag)
