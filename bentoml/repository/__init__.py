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

from bentoml import config


DEFAULT_LOCAL_REPO_PATH = config.get('repository', 'base_path')


# Bento, is a file format representing a saved BentoService
class BaseRepository(object):
    def add(self, bento_service_instance, version=None):
        pass

    def delete(self, bento_name, bento_version=None):
        pass

    def get(self, bento_name, bento_version):
        pass

    def list(self, bento_name):
        pass


class LocalRepository(BaseRepository):
    def __init__(self, base_path=DEFAULT_LOCAL_REPO_PATH):
        self.base_path = base_path

    def add(self, bento_service_instance, version=None):
        pass

    def delete(self, bento_name, bento_version=None):
        pass

    def get(self, bento_name, bento_version):
        pass

    def list(self, bento_name):
        pass


def get_local(base_path=DEFAULT_LOCAL_REPO_PATH):
    return LocalRepository(base_path)
