import logging
import uuid

import pytest

from bentoml.yatai.client import get_yatai_client
from bentoml.yatai.proto.repository_pb2 import BentoUri
from tests.bento_service_examples.example_bento_service import ExampleBentoService
from tests.yatai.local_yatai_service import local_yatai_service_container

logger = logging.getLogger('bentoml.test')

docker_image_tag = uuid.uuid4().hex[:8]


def test_sqlite_and_local_fs():
    with local_yatai_service_container(image_tag=docker_image_tag) as yatai_server_url:
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


@pytest.skip('Need to double check S3 credentials')
def test_yatai_server_with_sqlite_and_s3():
    # Note: Use pre-existing bucket instead of newly created bucket, because the
    # bucket's global DNS needs time to get set up.
    # https://github.com/boto/boto3/issues/1982#issuecomment-511947643
    s3_bucket_name = 's3://bentoml-e2e-test-repo/'

    with local_yatai_service_container(
        image_tag=docker_image_tag,
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
