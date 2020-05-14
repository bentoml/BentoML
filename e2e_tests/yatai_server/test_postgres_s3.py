import logging

from bentoml.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.yatai_server.utils import (
    get_bento_service,
    run_bento_service_prediction,
    start_yatai_server,
    modified_environ,
    BentoServiceForYataiTest,
)

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_postgres_and_s3(temporary_docker_postgres_url):
    # Note: Use pre-existing bucket instead of newly created bucket, because the
    # bucket's global DNS needs time to get set up.
    # https://github.com/boto/boto3/issues/1982#issuecomment-511947643

    s3_bucket_name = 's3://bentoml-e2e-test-repo/'

    with start_yatai_server(
        db_url=temporary_docker_postgres_url, repo_base_url=s3_bucket_name
    ) as yatai_service_url:
        logger.info(f'Setting config yatai_service.url to: {yatai_service_url}')
        with modified_environ(BENTOML__YATAI_SERVICE__URL=yatai_service_url):
            logger.info('Saving bento service')
            svc = BentoServiceForYataiTest()
            svc.save()
            bento_tag = f'{svc.name}:{svc.version}'
            logger.info('BentoService saved')

            logger.info("Display bentoservice info")
            get_svc_result = get_bento_service(svc.name, svc.version)
            logger.info(get_svc_result)
            assert (
                get_svc_result.bento.uri.type == BentoUri.LOCAL
            ), 'BentoService storage type mismatched, expect LOCAL'

            logger.info('Validate BentoService prediction result')
            run_result = run_bento_service_prediction(bento_tag, '[]')
            logger.info(run_result)
            assert 'cat' in run_result, 'Unexpected BentoService prediction result'

            logger.info('Delete BentoService for testing')
            delete_svc_result = delete_bento(bento_tag)
            logger.info(delete_svc_result)
            assert delete_svc_result is None, 'Unexpected delete BentoService message.'
