from bentoml.yatai.container import YataiContainer
from bentoml.yatai.repository.local_repository import LocalRepository
from bentoml.yatai.repository.s3_repository import S3Repository


def test_local_respository():
    container = YataiContainer(
        repository_type="file_system",
        file_system_directory="/Users/ssheng/bentoml/repository",
    )

    assert isinstance(container.repository(), LocalRepository)


# def test_s3_respository():
#     container = YataiContainer(
#         repository_type="s3", s3_base_url="s3://url", s3_endpoint_url=None,
#     )

#     assert isinstance(container.repository(), S3Repository)
