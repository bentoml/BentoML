import subprocess
import logging

from bentoml.proto.repository_pb2 import BentoUri
from e2e_tests.yatai_server.utils import (
    delete_bento_service,
    get_bento_service,
    run_bento_service_prediction,
    save_bento_service_with_channel_address,
    create_test_postgres,
    delete_test_postgres,
)

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_postgres_and_local_storage():
    logger.info('Setting yatai server channel address to BentoML config')
    proc, temp_dir, db_url = create_test_postgres()

    try:
        yatai_server_command = [
            'bentoml',
            'yatai-service-start',
            '--db-url',
            db_url,
            '--debug',
        ]
        logger.info(f'Running command {" ".join(yatai_server_command)}')
        proc = subprocess.Popen(
            yatai_server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        logger.info('Saving bento service')
        bento_name, bento_version = save_bento_service_with_channel_address()
        bento_tag = f'{bento_name}:{bento_version}'
        logger.info('BentoService saved')

        logger.info("Display bentoservice info")
        get_svc_result = get_bento_service(bento_tag)
        logger.info(get_svc_result)
        assert (
            get_svc_result.bento.uri.type == BentoUri.LOCAL
        ), 'BentoService storage type mismatched, expect LOCAL'

        logger.info('Validate BentoService prediction result')
        run_result = run_bento_service_prediction(bento_tag, '[]')
        logger.info(run_result)
        assert 'cat' in run_result, 'Unexpected BentoService prediction result'

        logger.info('Delete BentoService for testing')
        delete_svc_result = delete_bento_service(bento_tag)
        logger.info(delete_svc_result)
        assert (
            f'BentoService {bento_tag} deleted' in delete_svc_result
        ), 'Unexpected delete BentoService message.'

        logger.info('Display Yatai Server log')
        proc.terminate()
        server_std_out = proc.stdout.read().decode('utf-8')
        logger.info(server_std_out)
        logger.info('Shutting down YataiServer')
    finally:
        print(proc.stdout.read().decode('utf-8'))
        delete_test_postgres(proc, temp_dir)
