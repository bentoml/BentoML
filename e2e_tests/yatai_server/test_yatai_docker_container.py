import logging

from bentoml.yatai.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.yatai_server.utils import (
    modified_environ,
    BentoServiceForYataiTest,
    get_bento_service,
    run_bento_service_prediction,
)

logger = logging.getLogger('bentoml.test')


def test_docker_yatai_server_with_postgres(temporary_yatai_service_url):

    with modified_environ(BENTOML__YATAI_SERVICE__URL=temporary_yatai_service_url):
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
        assert f"{bento_tag} deleted" in delete_svc_result
