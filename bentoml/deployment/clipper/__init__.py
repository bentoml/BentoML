# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from bentoml.utils.exceptions import BentoMLException
from bentoml.deployment.utils import generate_bentoml_deployment_snapshot_path
from bentoml.deployment.base_deployment import Deployment
from bentoml.deployment.clipper.templates import DEFAULT_CLIPPER_DEPLOYER_PY


def check_clipper_availability():
    return


class ClipperDeployment(Deployment):
    """Clipper.ai Deployment manager.
    It creates a pure python function deployment in the clipper cluster
    """

    def __init__(self, archive_path, api_name=None):
        super(ClipperDeployment, self).__init__(archive_path)
        apis = self.bento_service.get_service_apis()
        if api_name:
            self.api = next(item for item in apis if item.name == api_name)
        elif len(apis) == 1:
            self.api = apis[0]
        else:
            raise BentoMLException(
                    'Please specify api-name, when more than one API is presented in the archive')

    def deploy(self):
        """Deploying bento service to clipper cluster.
        1. create snapshot location.
        2. call that deploy function in that snapshot
        """
        snapshot_path = generate_bentoml_deployment_snapshot_path(
            self.bento_service.name, self.bento_service.version, 'clipper')

        shutil.copytree(self.archive_path, snapshot_path)
        deployer_py_content = DEFAULT_CLIPPER_DEPLOYER_PY.format(
                class_name=self.bento_service.name)
        with open(os.path.join(snapshot_path, 'deployer.py'), 'w') as f:
            f.write(deployer_py_content)

        return

    def check_status(self):
        return

    def delete(self):
        return
