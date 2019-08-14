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


# Bento, is a file format representing a saved BentoService
@add_metaclass(ABCMeta)
class BaseRepository(object):
    @abstractmethod
    def add(self, bento_service_instance, version=None):
        """
        Adding a BentoService instance to target repository - this will resolve in a
        call to BentoService#save, which creates a new Bento that contains all
        serialized artifacts and related source code and configuration. Repository
        implementation will take care of storing and retriving the Bento files.
        """

    @abstractmethod
    def delete(self, bento_name, bento_version):
        """
        Deleting the Bento files that was added earlier
        """

    @abstractmethod
    def get(self, bento_name, bento_version=None):
        """
        Get a file path to the saved Bento files, path must be accessible form local
        machine either through NFS or pre-downloaded to local machine
        """

    @abstractmethod
    def list(self, bento_name=None):
        """
        List all saved versions of a given Bento name
        """


class LocalRepository(BaseRepository):
    def __init__(self, base_path):
        self.base_path = base_path

    def add(self, bento_service_instance, version=None):
        return archive.save(bento_service_instance, self.base_path, version=version)

    def delete(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        return shutil.rmtree(saved_path)

    def get_latest_version(self, bento_name):
        raise NotImplementedError

    def get(self, bento_name, bento_version=None):
        if bento_version is None or bento_version == 'latest':
            bento_version = self.get_latest_version(bento_name)

        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        return saved_path

    def list(self, bento_name=None):
        raise NotImplementedError


class S3Repository(BaseRepository):
    def __init__(self):
        raise NotImplementedError

    def add(self, bento_service_instance, version=None):
        raise NotImplementedError

    def delete(self, bento_name, bento_version=None):
        raise NotImplementedError

    def get(self, bento_name, bento_version=None):
        raise NotImplementedError

    def list(self, bento_name=None):
        raise NotImplementedError


def get_default_repository():
    default_repository_type = config.get('repository', 'default')

    if default_repository_type == 'local':
        return get_local_repository()
    elif default_repository_type == 's3':
        raise NotImplementedError("S3 repository is not yet implemented")
    else:
        raise ValueError(
            "Unknown default repository type: {}".format(default_repository_type)
        )


def get_local_repository(base_path=None):
    base_path = base_path or config.get('repository', 'base_path')
    return LocalRepository(base_path)
