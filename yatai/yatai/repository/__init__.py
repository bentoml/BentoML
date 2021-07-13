# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bentoml.yatai.repository.base_repository import BaseRepository
from bentoml.yatai.repository.file_system_repository import FileSystemRepository
from bentoml.yatai.repository.s3_repository import S3Repository
from bentoml.yatai.repository.gcs_repository import GCSRepository


def create_repository(
    repository_type: str,
    file_system_directory=None,
    s3_url=None,
    s3_endpoint_url=None,
    gcs_url=None,
) -> BaseRepository:
    """Creates a repository based on a provided type and parameters"""
    if repository_type == "s3":
        return S3Repository(s3_url, endpoint_url=s3_endpoint_url)
    elif repository_type == "gcs":
        return GCSRepository(gcs_url)
    elif repository_type == "file_system":
        return FileSystemRepository(file_system_directory)
    else:
        raise ValueError("Unrecognized repository type {}" % repository_type)
