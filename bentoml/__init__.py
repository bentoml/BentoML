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

from bentoml.configuration import inject_dependencies
from bentoml.utils.log import configure_logging

# Inject dependencies and configurations
inject_dependencies()

# Configuring logging properly before loading other modules
configure_logging()

from bentoml.saved_bundle import load_from_dir, save_to_dir  # noqa: E402
from bentoml.service import (  # noqa: E402
    BentoService,
    api_decorator as api,
    env_decorator as env,
    web_static_content_decorator as web_static_content,
    artifacts_decorator as artifacts,
    ver_decorator as ver,
    save,
)

from bentoml.cli import create_bentoml_cli

commandline_interface = create_bentoml_cli()

load = load_from_dir

__all__ = [
    "__version__",
    "api",
    "artifacts",
    "env",
    "web_static_content",
    "ver",
    "BentoService",
    # backward compatible
    "load",
    "load_from_dir",
    "save",
    "save_to_dir",
    "handlers",
    "adapters",
    "artifact",
]
