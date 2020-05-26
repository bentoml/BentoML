import logging

from bentoml.proto.repository_pb2 import BentoUri
from e2e_tests.cli_operations import delete_bento
from e2e_tests.utils import start_yatai_server, modified_environ
from e2e_tests.yatai_server.utils import (
    BentoServiceForYataiTest,
    get_bento_service,
    run_bento_service_prediction,
)

logger = logging.getLogger('bentoml.test')


grpc_port = 50054
ui_port = 3003
minio_env = {
    'AWS_ACCESS_KEY_ID': 'minioadmin',
    'AWS_SECRET_ACCESS_KEY': 'minioadmin',
    'AWS_REGION': 'us-east-1',
}


def test_yatai_server_with_sqlite_and_s3(
    yatai_service_docker_image_tag, minio_container_service
):
    # Note: Use pre-existing bucket instead of newly created bucket, because the
    # bucket's global DNS needs time to get set up.
    # https://github.com/boto/boto3/issues/1982#issuecomment-511947643

    s3_bucket_name = f's3://{minio_container_service["bucket_name"]}/'
    with start_yatai_server(
        docker_image=yatai_service_docker_image_tag,
        repo_base_url=s3_bucket_name,
        grpc_port=grpc_port,
        ui_port=ui_port,
        env=minio_env,
        s3_endpoint_url=minio_container_service['url'],
    ) as yatai_server_url:
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
                get_svc_result.bento.uri.type == BentoUri.S3
            ), 'BentoService storage type mismatched, expect S3'

            logger.info('Validate BentoService prediction result')
            run_result = run_bento_service_prediction(bento_tag, '[]')
            logger.info(run_result)
            assert 'cat' in run_result, 'Unexpected BentoService prediction result'

            logger.info('Delete BentoService for testing')
            delete_svc_result = delete_bento(bento_tag)
            logger.info(delete_svc_result)
            # expect_delete_message = f'BentoService {svc.name}:{svc.version} deleted\n'
            # assert (
            #     expect_delete_message == delete_svc_result
            # ), 'Unexpected delete BentoService message'
