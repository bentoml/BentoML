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

from bentoml import config
from bentoml.utils.s3 import is_s3_url
from bentoml.yatai.repository.base_repository import BaseRepository
from bentoml.yatai.repository.local_repository import LocalRepository
from bentoml.yatai.repository.s3_repository import S3Repository


class Repository(BaseRepository):
    def __init__(self, base_url=None, s3_endpoint_url=None):
        """
        :param base_url: either a local file system path or a s3-compatible path such as
            s3://my-bucket/some-prefix/
        :param s3_endpoint_url: configuring S3Repository to talk to a specific s3
            endpoint
        """

        if base_url is None:
            base_url = config().get('default_repository_base_url')

        if is_s3_url(base_url):
            self._repo = S3Repository(base_url, s3_endpoint_url)
        else:
            self._repo = LocalRepository(base_url)

    def add(self, bento_name, bento_version):
        return self._repo.add(bento_name, bento_version)

    def get(self, bento_name, bento_version):
        return self._repo.get(bento_name, bento_version)

    def dangerously_delete(self, bento_name, bento_version):
        return self._repo.dangerously_delete(bento_name, bento_version)
