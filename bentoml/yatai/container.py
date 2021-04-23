from dependency_injector import containers, providers

from bentoml.yatai.db import DB
from bentoml.yatai.repository.local_repository import LocalRepository
from bentoml.yatai.repository.s3_repository import S3Repository
from bentoml.yatai.repository.gcs_repository import GCSRepository


class YataiContainer(containers.DeclarativeContainer):

    file_system_directory = providers.Dependency()

    file_system_repository = providers.Factory(LocalRepository, file_system_directory,)

    s3_base_url = providers.Dependency()

    s3_endpoint_url = providers.Dependency()

    s3_signature_version = providers.Dependency()

    s3_repository = providers.Factory(S3Repository, s3_base_url, s3_endpoint_url,)

    gcs_base_url = providers.Dependency()

    gcs_repository = providers.Factory(GCSRepository, gcs_base_url,)

    repository_type = providers.Dependency()

    repository = providers.Selector(
        repository_type,
        file_system=file_system_repository,
        s3=s3_repository,
        gcs=gcs_repository,
    )

    database_url = providers.Dependency()

    database = providers.Factory(DB, database_url,)
