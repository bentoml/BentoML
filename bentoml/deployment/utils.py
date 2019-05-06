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
import json
import logging

from datetime import datetime

from bentoml.utils import Path


logger = logging.getLogger(__name__)


def generate_bentoml_deployment_snapshot_path(service_name, platform):
    return os.path.join(
        str(Path.home()), '.bentoml', 'deployment-snapshot', service_name, platform,
        datetime.now().isoformat())


def update_deployment_status(action, service_name, service_version, platform, additional_info):
    # 1. update the deployment status json
    # 2. log the user action into a log
    deployment_status_file_path = os.path.join(
        str(Path.home()), '.bentoml', 'deployment-status.json')
    deployment_log_path = os.path.join(str(Path.home()), '.bentoml', 'deployment-log')

    with open(deployment_status_file_path, 'wr') as f:
        data = json.loads(f)
        data[service_name][platform] = {
            "version": service_version,
            "created_at": datetime.now().isoformat(),
            "last_updated_at": datetime.now().isoformat(),
            "additional_info": additional_info
        }
        f.write(json.dumps(data))
    return
