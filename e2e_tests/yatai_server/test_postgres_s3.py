import logging

from bentoml.yatai.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.sample_bento_service import SampleBentoService
from e2e_tests.yatai_server.utils import (
    get_bento_service_info,
    execute_bentoml_run_command,
    local_yatai_server,
)

logger = logging.getLogger('bentoml.test')


def test_yatai_server_with_postgres_and_s3(postgres_db_container_url):
    # Note: Use pre-existing bucket instead of newly created bucket, because the
    # bucket's global DNS needs time to get set up.
    # https://github.com/boto/boto3/issues/1982#issuecomment-511947643

    s3_bucket_name = 's3://bentoml-e2e-test-repo/'

    with local_yatai_server(
        db_url=postgres_db_container_url, repo_base_url=s3_bucket_name
    ):
        logger.info('Saving bento service')
        svc = SampleBentoService()
        svc.save()
        bento_tag = f'{svc.name}:{svc.version}'
        logger.info('BentoService saved')

        logger.info("Display bentoservice info")
        get_svc_result = get_bento_service_info(svc.name, svc.version)
        logger.info(get_svc_result)
        assert (
            get_svc_result.bento.uri.type == BentoUri.S3
        ), 'BentoService storage type mismatched, expect S3'

        logger.info('Validate BentoService prediction result')
        run_result = execute_bentoml_run_command(bento_tag, '[]')
        assert 'cat' in run_result, 'Unexpected BentoService prediction result'

        logger.info(f'Deleting saved bundle {bento_tag}')
        delete_svc_result = delete_bento(bento_tag)
        assert f"{bento_tag} deleted" in delete_svc_result
