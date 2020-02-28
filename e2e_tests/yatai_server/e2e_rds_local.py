#!/usr/bin/env python

import subprocess
import logging
import sys
import traceback

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

if __name__ == '__main__':
    e2e_test_failed = False

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
        if get_svc_result.bento.uri.type != BentoUri.LOCAL:
            logger.error('Get bento service info failed')
            e2e_test_failed = True

        logger.info('Validate BentoService prediction result')
        run_result = run_bento_service_prediction(bento_tag, '[]')
        logger.info(run_result)
        if 'cat' not in run_result:
            logger.error('Run prediction failed')
            e2e_test_failed = True

        logger.info('Delete BentoService for testing')
        delete_svc_result = delete_bento_service(bento_tag)
        logger.info(delete_svc_result)
        if f'BentoService {bento_tag} deleted' not in delete_svc_result:
            logger.error('Delete bento service failed')
            e2e_test_failed = True

        logger.info('Display Yatai Server log')
        proc.terminate()
        server_std_out = proc.stdout.read().decode('utf-8')
        logger.info(server_std_out)

        logger.info('Shutting down YataiServer')
        delete_test_postgres(proc, temp_dir)
    except Exception:
        e2e_test_failed = True
        delete_test_postgres(proc, temp_dir)
        traceback.print_exc()

    if e2e_test_failed:
        logger.info('E2E YataiServer with Postgres and local fs failed')
        sys.exit(1)
    else:
        logger.info('E2E YataiServer with Postgres and local fs testing is successful')
        sys.exit(0)
