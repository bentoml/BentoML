# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

# From versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# from bentoml._internal.configuration import inject_dependencies
# from bentoml._internal.utils.log import configure_logging
#
# # Inject dependencies and configurations
# inject_dependencies()
#
# # Configuring logging properly before loading other modules
# configure_logging()

from ._internal.artifacts import ModelArtifact as BaseModel
from ._internal.artifacts import PickleArtifact
from ._internal.bundle import containerize, load
from ._internal.environment import env
from ._internal.inference_api import api, batch_api
from ._internal.repository import delete, get, list, pull, push
from ._internal.server import serve
from ._internal.service import Service
from ._internal.yatai_client import YataiClient

__all__ = [
    "__version__",
    "YataiClient",
    "PickleArtifact",
    "BaseModel",
    "list",
    "get",
    "delete",
    "push",
    "pull",
    "containerize",
    "serve",
    "Service",
    "env",
    "load",
    "api",
    "batch_api",
]
