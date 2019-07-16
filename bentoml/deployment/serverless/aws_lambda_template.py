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

import logging
import os

from ruamel.yaml import YAML

from bentoml.utils import Path

DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_AWS_DEPLOY_STAGE = "dev"

logger = logging.getLogger(__name__)

AWS_HANDLER_PY_TEMPLATE_HEADER = """\
try:
    import unzip_requirements
except ImportError:
    pass

from {class_name} import {class_name}

bento_service = {class_name}.load()
apis = bento_service.get_service_apis()

"""

AWS_FUNCTION_TEMPLATE = """\
def {api_name}(event, context):
    api = next(item for item in apis if item.name == '{api_name}')

    result = api.handle_aws_lambda_event(event)
    return result

"""


def generate_serverless_configuration_for_aws(
    service_name, apis, output_path, region, stage
):
    config_path = os.path.join(output_path, "serverless.yml")
    yaml = YAML()
    with open(config_path, "r") as f:
        content = f.read()
    serverless_config = yaml.load(content)

    serverless_config["service"] = service_name
    serverless_config["provider"]["region"] = region
    logger.info("Using user AWS region: %s", region)

    serverless_config["provider"]["stage"] = stage
    logger.info("Using AWS stage: %s", stage)

    serverless_config["functions"] = {}
    for api in apis:
        function_config = {
            "handler": "handler.{name}".format(name=api.name),
            "events": [
                {"http": {"path": "/{name}".format(name=api.name), "method": "post"}}
            ],
        }
        serverless_config["functions"][api.name] = function_config

    custom_config = {
        "apigwBinary": ["image/jpg", "image/jpeg", "image/png"],
        "pythonRequirements": {
            "useDownloadCache": True,
            "useStaticCache": True,
            "dockerizePip": True,
            "slim": True,
            "strip": True,
            "zip": True,
        },
    }

    serverless_config["custom"] = custom_config

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_handler_py(bento_service, apis, output_path):
    handler_py_content = AWS_HANDLER_PY_TEMPLATE_HEADER.format(
        class_name=bento_service.name
    )

    for api in apis:
        api_content = AWS_FUNCTION_TEMPLATE.format(api_name=api.name)
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, "handler.py"), "w") as f:
        f.write(handler_py_content)
    return


def create_aws_lambda_bundle(bento_service, output_path, region, stage):
    apis = bento_service.get_service_apis()
    generate_handler_py(bento_service, apis, output_path)
    generate_serverless_configuration_for_aws(
        bento_service.name, apis, output_path, region, stage
    )
    return
