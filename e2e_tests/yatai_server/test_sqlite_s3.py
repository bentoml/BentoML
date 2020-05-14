import logging

from bentoml.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.yatai_server.utils import (
    start_yatai_server,
    modified_environ,
    BentoServiceForYataiTest,
    get_bento_service,
    run_bento_service_prediction,
)

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_sqlite_and_s3():
    s3_bucket_name = 's3://bentoml-e2e-test-repo/'

    with start_yatai_server(repo_base_url=s3_bucket_name) as yatai_server_url:
        logger.info(f'Setting config yatai_service.url to: {yatai_server_url}')
        with modified_environ(BENTOML__YATAI_SERVICE__URL=yatai_server_url):
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
