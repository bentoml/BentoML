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

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from bentoml._internal.configuration import inject_dependencies
from bentoml._internal.utils.log import configure_logging

# Inject dependencies and configurations
inject_dependencies()

# Configuring logging properly before loading other modules
configure_logging()

from bentoml._internal.bundle import load_from_dir, save_to_dir  # noqa: E402
from bentoml._internal.service import BentoService
from bentoml._internal.service import api_decorator as api  # noqa: E402
from bentoml._internal.service import artifacts_decorator as artifacts
from bentoml._internal.service import env_decorator as env
from bentoml._internal.service import save
from bentoml._internal.service import ver_decorator as ver
from bentoml._internal.service import web_static_content_decorator as web_static_content

load = load_from_dir

cmd = [
    "list",
    "get",
    "delete",
    "push",
    "pull",
    "containerize",
    "serve"
]

sdk = [
    "Service",
    "env",
    "bundle",
    "load",
    "api",
    "batch_api",
]

__all__ = [
    "__version__",
] + sdk + cmd