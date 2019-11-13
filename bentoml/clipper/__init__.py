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
import shutil
import re
import logging

import docker

from bentoml.utils.tempdir import TempDirectory
from bentoml.archive import load_bento_service_metadata
from bentoml.deployment.utils import (
    process_docker_api_line,
    ensure_docker_available_or_raise,
)
from bentoml.handlers.clipper_handler import HANDLER_TYPE_TO_INPUT_TYPE
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)


CLIPPER_ENTRY = """\
from __future__ import print_function

import rpc # this is clipper's rpc.py module
import os
import sys

from bentoml import load_service_api

IMPORT_ERROR_RETURN_CODE = 3

api = load_service_api('/container/bento', '{api_name}')


class BentoServiceContainer(rpc.ModelContainerBase):

    def predict_ints(self, inputs):
        preds = api.handle_request(inputs)
        return [str(p) for p in preds]

    def predict_floats(self, inputs):
        preds = api.handle_request(inputs)
        return [str(p) for p in preds]

    def predict_doubles(self, inputs):
        preds = api.handle_request(inputs)
        return [str(p) for p in preds]

    def predict_bytes(self, inputs):
        preds = api.handle_request(inputs)
        return [str(p) for p in preds]

    def predict_strings(self, inputs):
        preds = api.handle_request(inputs)
        return [str(p) for p in preds]


if __name__ == "__main__":
    print("Starting BentoService Clipper Containter")
    rpc_service = rpc.RPCService()

    try:
        model = BentoServiceContainer()
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)

    rpc_service.start(model)
"""


CLIPPER_DOCKERFILE = """\
FROM clipper/python36-closure-container:0.4.1

# copy over model files
COPY . /container
WORKDIR /container

# Install pip dependencies
RUN pip install -r /container/bento/requirements.txt

# run user defined setup script
RUN if [ -f /container/bento/setup.sh ]; then /bin/bash -c /container/bento/setup.sh; fi

ENV CLIPPER_MODEL_NAME={model_name}
ENV CLIPPER_MODEL_VERSION={model_version}

# Stop running entry point from base image
ENTRYPOINT []
# Run BentoML bundle for clipper
CMD ["python", "/container/clipper_entry.py"]
"""  # noqa: E501



def get_clipper_compatiable_string(item):
    """Generate clipper compatiable string. It must be a valid DNS-1123.
    It must consist of lower case alphanumeric characters, '-' or '.',
    and must start and end with an alphanumeric character

    :param item: String
    :return: string
    """

    pattern = re.compile("[^a-zA-Z0-9-]")
    result = re.sub(pattern, "-", item)
    return result.lower()


def deploy_bentoml(clipper_conn, bundle_path, api_name, model_name=None, labels=None):
    """Deploy bentoml bundle to clipper cluster

    Args:
        clipper_conn(clipper_admin.ClipperConnection): Clipper connection instance
        bundle_path(str): Path to the bentoml service archive.
        api_name(str): name of the api that will be used as prediction function for
            clipper cluster
        model_name(str): Model's name for clipper cluster
        labels(:obj:`list(str)`, optional): labels for clipper model

    Returns:
        tuple: Model name and model version that deployed to clipper

    """
    ensure_docker_available_or_raise() # docker is required to build clipper model image

    if not clipper_conn.connected:
        raise BentoMLException(
            "No connection to Clipper cluster. CallClipperConnection.connect to "
            "connect to an existing cluster or ClipperConnnection.start_clipper to "
            "create a new one"
        )

    bento_service_metadata = load_bento_service_metadata(bundle_path)

    try:
        api_metadata = next((api for api in bento_service_metadata.apis))
    except StopIteration:
        raise BentoMLException(
            "Can't find API '{}' in BentoService bundle {}".format(
                api_name, bento_service_metadata.name
            )
        )

    if api_metadata.handler_type not in HANDLER_TYPE_TO_INPUT_TYPE:
        raise BentoMLException(
            "Only BentoService APIs using ClipperHandler can be deployed to Clipper"
        )

    input_type = HANDLER_TYPE_TO_INPUT_TYPE[api_metadata.handler_type]
    model_name = model_name or get_clipper_compatiable_string(
        bento_service_metadata.name + "-" + api_metadata.name
    )
    model_version = get_clipper_compatiable_string(bento_service_metadata.version)

    with TempDirectory() as tempdir:
        entry_py_content = CLIPPER_ENTRY.format(api_name=api_name)
        model_path = os.path.join(tempdir, "bento")
        shutil.copytree(bundle_path, model_path)

        with open(os.path.join(tempdir, "clipper_entry.py"), "w") as f:
            f.write(entry_py_content)

        docker_content = CLIPPER_DOCKERFILE.format(
            model_name=model_name, model_version=model_version
        )
        with open(os.path.join(tempdir, "Dockerfile-clipper"), "w") as f:
            f.write(docker_content)

        docker_api = docker.APIClient()
        image_tag = "clipper-model-{}:{}".format(
            bento_service_metadata.name.lower(), bento_service_metadata.version
        )

        for line in docker_api.build(
            path=tempdir, dockerfile="Dockerfile-clipper", tag=image_tag
        ):
            process_docker_api_line(line)

        logger.info(
            "Successfully built docker image %s for Clipper deployment", image_tag
        )

    clipper_conn.deploy_model(
        name=model_name,
        version=model_version,
        input_type=input_type,
        image=image_tag,
        labels=labels,
    )

    return model_name, model_version
