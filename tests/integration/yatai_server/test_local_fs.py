import logging
import time
import pytest

from bentoml.yatai.client import get_yatai_client
from bentoml.yatai.proto.repository_pb2 import BentoUri
from tests.bento_services.example_bento_service import ExampleBentoService
from tests.integration.yatai_server.utils import (
    yatai_service_container,
    local_yatai_service_from_cli,
)


logger = logging.getLogger("bentoml.test")


def test_sqlite_and_local_fs():
    with yatai_service_container() as yatai_server_url:
        yc = get_yatai_client(yatai_server_url)
        svc = ExampleBentoService()
        svc.pack("model", [1, 2, 3])
        bento_tag = f"{svc.name}:{svc.version}"
        logger.info(f"Saving BentoML saved bundle {bento_tag}")
        svc.save(yatai_url=yatai_server_url)

        bento_pb = yc.repository.get(bento_tag)
        assert (
            bento_pb.uri.type == BentoUri.LOCAL
        ), "BentoService storage type mismatched, expect LOCAL"

        logger.info(f"Deleting saved bundle {bento_tag}")
        delete_svc_result = yc.repository.delete(bento_tag)
        assert delete_svc_result is None


@pytest.mark.skip("Skipping Postgres test on Github Action as it continues been flaky")
def test_yatai_server_with_postgres_and_local_storage():
    postgres_db_url = "postgresql://postgres:postgres@localhost/bentoml:5432"

    from sqlalchemy_utils import create_database

    create_database(postgres_db_url)
    time.sleep(60)

    with local_yatai_service_from_cli(db_url=postgres_db_url) as yatai_server_url:
        logger.info("Saving bento service")
        logger.info(f"yatai url is {yatai_server_url}")
        svc = ExampleBentoService()
        svc.pack("model", [1, 2, 3])
        bento_tag = f"{svc.name}:{svc.version}"
        logger.info(f"Saving BentoML saved bundle {bento_tag}")
        svc.save(yatai_url=yatai_server_url)

        yc = get_yatai_client(yatai_server_url)
        bento_pb = yc.repository.get(bento_tag)
        assert (
            bento_pb.uri.type == BentoUri.LOCAL
        ), "BentoService storage type mismatched, expect LOCAL"
