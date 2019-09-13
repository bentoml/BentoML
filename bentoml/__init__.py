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

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from bentoml.config import load_config

config = load_config()

from bentoml.utils.log import configure_logging

configure_logging()

from bentoml import archive
from bentoml import handlers
from bentoml.service import (
    BentoService,
    api_decorator as api,
    env_decorator as env,
    artifacts_decorator as artifacts,
    ver_decorator as ver,
)
from bentoml.server import metrics
from bentoml.utils.log import configure_logging
from bentoml.yatai.python_api import upload_bento_service

load = archive.load
save_to_dir = archive.save_to_dir
save = upload_bento_service


__all__ = [
    "__version__",
    "api",
    "artifacts",
    "config",
    "env",
    "ver",
    "metrics",
    "BentoService",
    "handlers",
    "load",
    "save",
    "save_to_dir",
]
