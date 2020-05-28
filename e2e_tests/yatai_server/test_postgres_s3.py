import pytest
import logging

from bentoml.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.utils import start_yatai_server, modified_environ
from e2e_tests.yatai_server.utils import (
    get_bento_service,
    run_bento_service_prediction,
    BentoServiceForYataiTest,
)

logger = logging.getLogger('bentoml.test')

# MinIO bucket
s3_bucket_name = 's3://bentoml-e2e-test-repo/'
grpc_port = 50055
ui_port = 3004
minio_env = {
    'AWS_ACCESS_KEY_ID': 'minioadmin',
    'AWS_SECRET_ACCESS_KEY': 'minioadmin',
    'AWS_REGION': 'us-east-1',
}

@pytest.mark.skip(
    reason='need more investigation on two docker containers not play nicely together'
)
def test_yatai_server_with_postgres_and_s3(
    yatai_service_docker_image_tag, minio_container_service, postgres_docker_container
):
    s3_bucket_name = f's3://{minio_container_service["bucket_name"]}/'
    s3_endpoint_url = minio_container_service['url'].replace('127.0.0.1', 's3-container')

    with start_yatai_server(
        docker_image=yatai_service_docker_image_tag,
        db_url=postgres_docker_container['url'],
        db_container_name=postgres_docker_container['container_name'],
        grpc_port=grpc_port,
        ui_port=ui_port,
        env=minio_env,
        repo_base_url=s3_bucket_name,
        s3_container_name=minio_container_service['container_name'],
        # s3_endpoint_url=minio_container_service['url'],
        s3_endpoint_url=s3_endpoint_url
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
                get_svc_result.bento.uri.type == BentoUri.S3
            ), 'BentoService storage type mismatched, expect S3'

            logger.info('Validate BentoService prediction result')
            run_result = run_bento_service_prediction(bento_tag, '[]')
            logger.info(run_result)
            assert 'cat' in run_result, 'Unexpected BentoService prediction result'

            logger.info('Delete BentoService for testing')
            delete_svc_result = delete_bento(bento_tag)
            logger.info(delete_svc_result)
            # expect_delete_message = (
            #     f'BentoService {svc.name}:{svc.version} deleted\n'
            # )
            # assert (
            #     expect_delete_message == delete_svc_result
            # ), 'Unexpected delete BentoService message'
