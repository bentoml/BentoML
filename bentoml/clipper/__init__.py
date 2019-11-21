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
from bentoml.bundler import load_bento_service_metadata
from bentoml.deployment.utils import (
    process_docker_api_line,
    ensure_docker_available_or_raise,
)
from bentoml.handlers.clipper_handler import HANDLER_TYPE_TO_INPUT_TYPE
from bentoml.exceptions import BentoMLException
from bentoml.utils.usage_stats import track

logger = logging.getLogger(__name__)


CLIPPER_ENTRY = """\
from __future__ import print_function

import rpc # this is clipper's rpc.py module
import os
import sys

from bentoml.bundler import load_bento_service_api

IMPORT_ERROR_RETURN_CODE = 3


class BentoServiceContainer(rpc.ModelContainerBase):

    def __init__(self, bentoml_bundle_path, api_name):
        bento_service_api = load_bento_service_api(bentoml_bundle_path, api_name)
        self.predict_func = bento_service_api._func

    def predict_ints(self, inputs):
        preds = self.predict_func(inputs)
        return [str(p) for p in preds]

    def predict_floats(self, inputs):
        preds = self.predict_func(inputs)
        return [str(p) for p in preds]

    def predict_doubles(self, inputs):
        preds = self.predict_func(inputs)
        return [str(p) for p in preds]

    def predict_bytes(self, inputs):
        preds = self.predict_func(inputs)
        return [str(p) for p in preds]

    def predict_strings(self, inputs):
        preds = self.predict_func(inputs)
        return [str(p) for p in preds]


if __name__ == "__main__":
    print("Starting BentoService Clipper Containter")
    rpc_service = rpc.RPCService()

    try:
        model = BentoServiceContainer('/container/bento', '{api_name}')
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)

    rpc_service.start(model)
"""


CLIPPER_DOCKERFILE = """\
FROM {base_image}

# copy over model files
COPY . /container
WORKDIR /container

# Install pip dependencies
RUN pip install --upgrade numpy && pip install -r /container/bento/requirements.txt

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /container/bento/bentoml_init.sh ]; then cd /container/bento && /bin/bash -c /container/bento/bentoml_init.sh; fi

# run user defined setup script
RUN if [ -f /container/bento/setup.sh ]; then cd /container/bento && /bin/bash -c /container/bento/setup.sh; fi

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
        bundle_path(str): Path to the saved BentomlService bundle.
        api_name(str): name of the api that will be used as prediction function for
            clipper cluster
        model_name(str): Model's name for clipper cluster
        labels(:obj:`list(str)`, optional): labels for clipper model

    Returns:
        tuple: Model name and model version that deployed to clipper

    """
    track("clipper-deploy", {'bento_service_bundle_path': bundle_path})

    # docker is required to build clipper model image
    ensure_docker_available_or_raise()

    if not clipper_conn.connected:
        raise BentoMLException(
            "No connection to Clipper cluster. CallClipperConnection.connect to "
            "connect to an existing cluster or ClipperConnnection.start_clipper to "
            "create a new one"
        )

    bento_service_metadata = load_bento_service_metadata(bundle_path)

    try:
        api_metadata = next(
            (api for api in bento_service_metadata.apis if api.name == api_name)
        )
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

        if bento_service_metadata.env.python_version.startswith("3.6"):
            base_image = "clipper/python36-closure-container:0.4.1"
        elif bento_service_metadata.env.python_version.startswith("2.7"):
            base_image = "clipper/python-closure-container:0.4.1"
        else:
            raise BentoMLException(
                "Python version {} is not supported in Clipper".format(
                    bento_service_metadata.env.python_version
                )
            )

        docker_content = CLIPPER_DOCKERFILE.format(
            model_name=model_name, model_version=model_version, base_image=base_image
        )
        with open(os.path.join(tempdir, "Dockerfile-clipper"), "w") as f:
            f.write(docker_content)

        docker_api = docker.APIClient()
        clipper_model_docker_image_tag = "clipper-model-{}:{}".format(
            bento_service_metadata.name.lower(), bento_service_metadata.version
        )
        for line in docker_api.build(
            path=tempdir,
            dockerfile="Dockerfile-clipper",
            tag=clipper_model_docker_image_tag,
        ):
            process_docker_api_line(line)

        logger.info(
            "Successfully built docker image %s for Clipper deployment",
            clipper_model_docker_image_tag,
        )

    clipper_conn.deploy_model(
        name=model_name,
        version=model_version,
        input_type=input_type,
        image=clipper_model_docker_image_tag,
        labels=labels,
    )

    track("clipper-deploy-success", {'bento_service_bundle_path': bundle_path})
    return model_name, model_version
