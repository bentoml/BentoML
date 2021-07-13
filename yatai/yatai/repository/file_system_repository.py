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

import os
import shutil
import logging
from pathlib import Path

from bentoml.exceptions import YataiRepositoryException
from bentoml.yatai.proto.repository_pb2 import BentoUri
from bentoml.yatai.repository.base_repository import BaseRepository


logger = logging.getLogger(__name__)


class FileSystemRepository(BaseRepository):
    def __init__(self, base_url):
        """
        :param base_url: local file system path that will be used as the root directory
            of this saved bundle repository
        """
        if not os.path.exists(base_url):
            # make sure local repo base path exist
            os.mkdir(base_url)

        self.base_path = base_url
        self.uri_type = BentoUri.LOCAL

    def add(self, bento_name, bento_version):
        # Full path containing saved BentoService bundle, it the base path with service
        # name and service version as prefix. e.g.:
        # with base_path = '/tmp/my_bento_repo/', the saved bento will resolve in
        # the directory: '/tmp/my_bento_repo/service_name/version/'
        target_dir = os.path.join(self.base_path, bento_name, bento_version)

        # Ensure parent directory exist
        Path(os.path.join(self.base_path), bento_name).mkdir(
            parents=True, exist_ok=True
        )

        # Raise if target bento version already exist in storage
        if os.path.exists(target_dir):
            raise YataiRepositoryException(
                "Existing BentoService bundle {name}:{version} found in repository: "
                "{target_dir}".format(
                    name=bento_name, version=bento_version, target_dir=target_dir
                )
            )

        return BentoUri(type=self.uri_type, uri=target_dir)

    def get(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        if not os.path.exists(saved_path):
            logger.warning(
                "BentoML bundle %s:%s not found in target repository",
                bento_name,
                bento_version,
            )
        return saved_path

    def dangerously_delete(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        try:
            return shutil.rmtree(saved_path)
        except FileNotFoundError:
            logger.warning(
                "BentoService %s:%s has already been deleted from local storage",
                bento_name,
                bento_version,
            )
            return
