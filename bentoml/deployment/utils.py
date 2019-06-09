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

from datetime import datetime

from bentoml.config import BENTOML_HOME


def generate_bentoml_deployment_snapshot_path(service_name, service_version, platform):
    return os.path.join(
        BENTOML_HOME,
        "deployment-snapshots",
        platform,
        service_name,
        service_version,
        datetime.now().isoformat(),
    )
