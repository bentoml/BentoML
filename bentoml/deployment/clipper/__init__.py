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

import docker

from bentoml.archive import load
from bentoml.handlers import ImageHandler
from bentoml.deployment.utils import (
    generate_bentoml_deployment_snapshot_path,
    process_docker_api_line,
)
from bentoml.deployment.clipper.templates import (
    DEFAULT_CLIPPER_ENTRY,
    DOCKERFILE_CLIPPER,
)
from bentoml.exceptions import BentoMLException


def generate_clipper_compatiable_string(item):
    """Generate clipper compatiable string. It must be a valid DNS-1123.
    It must consist of lower case alphanumeric characters, '-' or '.',
    and must start and end with an alphanumeric character

    :param item: String
    :return: string
    """

    pattern = re.compile("[^a-zA-Z0-9-]")
    result = re.sub(pattern, "-", item)
    return result.lower()


def deploy_bentoml(
    clipper_conn,
    archive_path,
    api_name,
    input_type="strings",
    model_name=None,
    labels=["bentoml"],
):
    """Deploy bentoml bundle to clipper cluster

    Args:
        clipper_conn(clipper_admin.ClipperConnection): Clipper connection instance
        archive_path(str): Path to the bentoml service archive.
        api_name(str): name of the api that will be used as prediction function for
            clipper cluster
        input_type(str): Input type that clipper accept. The default input_type for
            image handler is `bytes`, for other handlers is `strings`. Availabel input_type
            are `integers`, `floats`, `doubles`, `bytes`, or `strings`
        model_name(str): Model's name for clipper cluster
        labels(:obj:`list(str)`, optional): labels for clipper model

    Returns:
        tuple: Model name and model version that deployed to clipper

    """
    bento_service = load(archive_path)
    apis = bento_service.get_service_apis()

    if api_name:
        api = next(item for item in apis if item.name == api_name)
    elif len(apis) == 1:
        api = apis[0]
    else:
        raise BentoMLException(
            "Please specify api-name, when more than one API is present in the archive"
        )
    model_name = model_name or generate_clipper_compatiable_string(
        bento_service.name + "-" + api.name
    )
    version = generate_clipper_compatiable_string(bento_service.version)

    if isinstance(api.handler, ImageHandler):
        input_type = "bytes"

    try:
        clipper_conn.start_clipper()
    except docker.errors.APIError:
        clipper_conn.connect()
    except Exception:
        raise BentoMLException("Can't start or connect with clipper cluster")

    snapshot_path = generate_bentoml_deployment_snapshot_path(
        bento_service.name, bento_service.version, "clipper"
    )

    entry_py_content = DEFAULT_CLIPPER_ENTRY.format(
        api_name=api.name, input_type=input_type
    )
    model_path = os.path.join(snapshot_path, "bento")
    shutil.copytree(archive_path, model_path)

    with open(os.path.join(snapshot_path, "clipper_entry.py"), "w") as f:
        f.write(entry_py_content)

    docker_content = DOCKERFILE_CLIPPER.format(
        model_name=model_name, model_version=version
    )
    with open(os.path.join(snapshot_path, "Dockerfile-clipper"), "w") as f:
        f.write(docker_content)

    docker_api = docker.APIClient()
    image_tag = bento_service.name.lower() + "-clipper:" + bento_service.version
    for line in docker_api.build(
        path=snapshot_path, dockerfile="Dockerfile-clipper", tag=image_tag
    ):
        process_docker_api_line(line)

    clipper_conn.deploy_model(
        name=model_name,
        version=version,
        input_type=input_type,
        image=image_tag,
        labels=labels,
    )

    return model_name, version
