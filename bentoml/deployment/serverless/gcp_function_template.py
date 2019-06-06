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
import logging

from ruamel.yaml import YAML

from bentoml.utils import Path

DEFAULT_GCP_REGION = "us-west2"
DEFAULT_GCP_DEPLOY_STAGE = "dev"

logger = logging.getLogger(__name__)

GOOGLE_MAIN_PY_TEMPLATE_HEADER = """\
from {class_name} import {class_name}

bento_service = {class_name}.load()
apis = bento_service.get_service_apis()

"""

GOOGLE_FUNCTION_TEMPLATE = """\
def {api_name}(request):
    api = next(item for item in apis if item.name == '{api_name}')

    result = api.handle_request(request)
    return result

"""


def generate_serverless_configuration_for_google(
    bento_service, apis, output_path, region, stage
):
    config_path = os.path.join(output_path, "serverless.yml")
    yaml = YAML()
    with open(config_path, "r") as f:
        content = f.read()
    serverless_config = yaml.load(content)

    serverless_config["service"] = bento_service.name
    serverless_config["provider"]["project"] = bento_service.name

    serverless_config["provider"]["region"] = region
    logger.info("Using user defined Google region: %s", region)

    serverless_config["provider"]["stage"] = stage
    logger.info("Using user defined Google stage: %s", stage)

    serverless_config["functions"] = {}
    for api in apis:
        function_config = {"handler": api.name, "events": [{"http": "path"}]}
        serverless_config["functions"][api.name] = function_config

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_main_py(bento_service, apis, output_path):
    handler_py_content = GOOGLE_MAIN_PY_TEMPLATE_HEADER.format(
        class_name=bento_service.name
    )

    for api in apis:
        api_content = GOOGLE_FUNCTION_TEMPLATE.format(api_name=api.name)
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, "main.py"), "w") as f:
        f.write(handler_py_content)
    return


def create_gcp_function_bundle(bento_service, output_path, region, stage):
    apis = bento_service.get_service_apis()
    generate_main_py(bento_service, apis, output_path)
    generate_serverless_configuration_for_google(
        bento_service, apis, output_path, region, stage
    )
    return
