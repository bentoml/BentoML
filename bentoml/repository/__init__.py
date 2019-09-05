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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from six import add_metaclass
from abc import abstractmethod, ABCMeta

from bentoml import config
from bentoml import archive
from bentoml.exceptions import BentoMLRepositoryException
from bentoml.utils.s3 import is_s3_url
from bentoml.utils import Path
from bentoml.utils.usage_stats import track_save


@add_metaclass(ABCMeta)
class BentoRepositoryBase(object):
    """
    BentoRepository is the interface for managing saved Bentos over file system or
    cloud storage systems.

    A Bento is a BentoService serialized into a standard file format that can be
    easily load back to a Python session, installed as PyPI package, or run in Conda
    or docker environment with all dependencies configured automatically
    """

    @abstractmethod
    def add(self, bento_service):
        """
        Adding a BentoService instance to target repository - this will resolve in a
        call to BentoService#save_to_dir, which creates a new Bento that contains all
        serialized artifacts and related source code and configuration. Repository
        implementation will take care of storing and retriving the Bento files.
        Return value is an URL(file path or s3 path for example), pointing to the
        saved Bento file directory
        """

    @abstractmethod
    def get(self, bento_name, bento_version):
        """
        Get a file path to the saved Bento files, path must be accessible form local
        machine either through NFS or pre-downloaded to local machine
        """

    @abstractmethod
    def dangerously_delete(self, bento_name, bento_version):
        """
        Deleting the Bento files that was added to this repository earlier, this may
        break existing deployments or create issues when doing deployment rollback
        """


class _LocalBentoRepository(BentoRepositoryBase):
    def __init__(self, base_url):
        if not os.path.exists(base_url):
            raise BentoMLRepositoryException(
                "Local path '{}' not found when " "initializing local Bento repository"
            )

        self.base_path = base_url

    def add(self, bento_service):
        Path(os.path.join(self.base_path), bento_service.name).mkdir(
            parents=True, exist_ok=True
        )
        # Full path containing saved BentoArchive, it the base path with service name
        # and service version as prefix. e.g.:
        # with base_path = '/tmp/my_bento_archive/', the saved bento will resolve in
        # the directory: '/tmp/my_bento_archive/service_name/version/'
        target_dir = os.path.join(
            self.base_path, bento_service.name, bento_service.version
        )

        if os.path.exists(target_dir):
            raise BentoMLRepositoryException(
                "Existing Bento {name}:{version} found in archive: {target_dir}".format(
                    name=bento_service.name,
                    version=bento_service.version,
                    target_dir=target_dir,
                )
            )
        os.mkdir(target_dir)

        archive.save_to_dir(bento_service, target_dir)

        return target_dir

    def get(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        if not os.path.exists(saved_path):
            raise BentoMLRepositoryException(
                "Bento {}:{} not found in target repository".format(
                    bento_name, bento_version
                )
            )
        return saved_path

    def dangerously_delete(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        return shutil.rmtree(saved_path)


class _S3BentoRepository(BentoRepositoryBase):
    def __init__(self, base_url):
        self.base_url = base_url

    def add(self, bento_service):
        raise NotImplementedError

    def get(self, bento_name, bento_version):
        raise NotImplementedError

    def dangerously_delete(self, bento_name, bento_version):
        raise NotImplementedError


DEFAULT_REPOSITORY_BASE_URL = config.get('default_repository_base_url')


class BentoRepository(BentoRepositoryBase):
    def __init__(self, base_url=None):
        if base_url is None:
            base_url = DEFAULT_REPOSITORY_BASE_URL

        if is_s3_url(base_url):
            self._repo = _S3BentoRepository(base_url)
        else:
            self._repo = _LocalBentoRepository(base_url)

    def add(self, bento_service):
        return self._repo.add(bento_service)

    def get(self, bento_name, bento_version):
        return self._repo.get(bento_name, bento_version)

    def dangerously_delete(self, bento_name, bento_version):
        return self._repo.dangerously_delete(bento_name, bento_version)


def save(bento_service, base_path=None, version=None):
    if version is not None:
        bento_service.set_version(version)

    # TODO: Callding BentoRepository directly for now, this should be changed to
    # a yatei service call instead, which can be either a local service or RPC
    # client calling remote service
    repo = BentoRepository(base_path)
    return repo.add(bento_service)