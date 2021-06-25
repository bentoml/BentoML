import os

from bentoml.yatai.client import get_yatai_client
from tests.bento_service_examples.example_bento_service import ExampleBentoService
from tests.integration.yatai_server.utils import yatai_service_container


def test_push_and_pull():
    with yatai_service_container() as yatai_server_url:
        svc = ExampleBentoService()
        bento_tag = f'{svc.name}:{svc.version}'
        saved_path = svc.save()
        yc = get_yatai_client(yatai_server_url)

        pushed_path = yc.repository.push(bento_tag)
        assert pushed_path != saved_path

        local_yc = get_yatai_client()
        delete_result = local_yc.repository.delete(bento_tag)
        assert delete_result is None
        assert os.path.exists(saved_path) is False

        pull_result = yc.repository.pull(bento_tag)
        assert pull_result == saved_path
