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


from bentoml.archive import load


class LegacyDeployment(object):
    """LegacyDeployment is spec for describing what actions deployment should have
    to interact with BentoML cli and BentoML service archive.
    """

    def __init__(self, archive_path):
        self.bento_service = load(archive_path)
        self.archive_path = archive_path

    def deploy(self):
        """Deploy bentoml service.
        """
        raise NotImplementedError

    def check_status(self):
        """Check deployment status
        """
        raise NotImplementedError

    def delete(self):
        """Delete deployment, if deployment is active
        """
        raise NotImplementedError
