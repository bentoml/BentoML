import logging

from bentoml.yatai.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.sample_bento_service import SampleBentoService
from e2e_tests.yatai_server.utils import (
    get_bento_service_info,
    yatai_server_container,
)

logger = logging.getLogger('bentoml.test')


def test_docker_yatai_server_with_postgres():
    with yatai_server_container():
        svc = SampleBentoService()
        bento_tag = f'{svc.name}:{svc.version}'
        logger.info(f'Saving BentoML saved bundle {bento_tag}')
        svc.save()

        get_svc_result = get_bento_service_info(svc.name, svc.version)
        logger.info(f'Retrived BentoML saved bundle {bento_tag} info: {get_svc_result}')
        assert (
            get_svc_result.bento.uri.type == BentoUri.LOCAL
        ), 'BentoService storage type mismatched, expect LOCAL'

        # Loading from LocalRepository based remote YataiService is not yet supported
        # logger.info('Validate BentoService CLI prediction result')
        # run_result = run_bento_service_prediction(bento_tag, '[]')
        # assert 'cat' in run_result, f'Unexpected prediction result: {run_result}'

        logger.info(f'Deleting saved bundle {bento_tag}')
        delete_svc_result = delete_bento(bento_tag)
        assert f"{bento_tag} deleted" in delete_svc_result
