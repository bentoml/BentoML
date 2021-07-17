from typing import Any
from yatai.yatai.repository.base_repository import BaseRepository as BaseRepository
from yatai.yatai.repository.file_system_repository import FileSystemRepository as FileSystemRepository
from yatai.yatai.repository.gcs_repository import GCSRepository as GCSRepository
from yatai.yatai.repository.s3_repository import S3Repository as S3Repository

def create_repository(repository_type: str, file_system_directory: Any | None = ..., s3_url: Any | None = ..., s3_endpoint_url: Any | None = ..., gcs_url: Any | None = ...) -> BaseRepository: ...
