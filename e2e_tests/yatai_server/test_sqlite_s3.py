import logging

from bentoml.yatai.proto.repository_pb2 import BentoUri
from bentoml.yatai.client import get_yatai_client
from e2e_tests.sample_bento_service import SampleBentoService
from e2e_tests.yatai_server.utils import (
    local_yatai_server,
    execute_bentoml_run_command,
)

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_sqlite_and_s3():
    # Note: Use pre-existing bucket instead of newly created bucket, because the
    # bucket's global DNS needs time to get set up.
    # https://github.com/boto/boto3/issues/1982#issuecomment-511947643

    s3_bucket_name = 's3://bentoml-e2e-test-repo/'

    with local_yatai_server(repo_base_url=s3_bucket_name) as yatai_service_url:
        yc = get_yatai_client(yatai_service_url)
        logger.info('Saving bento service')
        svc = SampleBentoService()
        svc.save(yatai_url=yatai_service_url)
        bento_tag = f'{svc.name}:{svc.version}'
        logger.info('BentoService saved')

        logger.info("Display bentoservice info")
        bento = yc.repository.get(bento_tag)
        logger.info(bento)
        assert (
            bento.uri.type == BentoUri.S3
        ), 'BentoService storage type mismatched, expect S3'

        logger.info('Validate BentoService prediction result')
        run_result = execute_bentoml_run_command(
            bento_tag=bento_tag, data='[]', yatai_url=yatai_service_url
        )
        logger.info(run_result)
        assert 'cat' in run_result, 'Unexpected BentoService prediction result'

        logger.info(f'Deleting saved bundle {bento_tag}')
        yc.repository.delete(bento_tag=bento_tag)
