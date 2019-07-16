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

from bentoml import handlers
from bentoml.config import config
from bentoml.version import __version__

from bentoml.service import (
    BentoService,
    api_decorator as api,
    env_decorator as env,
    artifacts_decorator as artifacts,
    ver_decorator as ver,
)
from bentoml.server import metrics
from bentoml.archive import save, load

from bentoml.utils.log import configure_logging
from bentoml import deployment

configure_logging()

__all__ = [
    "__version__",
    "api",
    "artifacts",
    "config",
    "env",
    "ver",
    "save",
    "load",
    "handlers",
    "metrics",
    "BentoService",
    "deployment",
]
