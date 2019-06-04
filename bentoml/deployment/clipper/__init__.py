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

import docker

from bentoml.archive import load
from bentoml.handlers import ImageHandler
from bentoml.deployment.utils import generate_bentoml_deployment_snapshot_path, \
    process_docker_api_line
from bentoml.deployment.clipper.templates import DEFAULT_CLIPPER_ENTRY, DOCKERFILE_CLIPPER
from bentoml.utils.exceptions import BentoMLException


def build_docker_image(snapshot_path):
    docker_api = docker.APIClient()
    tag = "something"
    for line in docker_api.build(path=snapshot_path, dockerfile='Dockerfile-clipper', tag=tag):
        process_docker_api_line(line)
    return tag


def deploy_bentoml(clipper_conn, archive_path, api_name):
    """Deploy bentoml bundle to clipper cluster

    :param clipper_conn: Clipper connection.
    :param archive_path: String, Path to bentoml bundle, it could be local filepath or s3 path
    :param api_name: String, Name of the api that will be running in the clipper cluster
    """
    bento_service = load(archive_path)
    apis = bento_service.get_service_apis()

    if api_name:
        api = next(item for item in apis if item.name == api_name)
    elif len(apis) == 1:
        api = apis[0]
    else:
        raise BentoMLException(
            'Please specify api-name, when more than one API is present in the archive')
    application_name = bento_service.name + '-' + api.name
    input_type = 'strings'

    if isinstance(api.handler, ImageHandler):
        input_type = 'bytes'

    try:
        clipper_conn.start_clipper()
    except docker.errors.APIError:
        clipper_conn.connect()
    except Exception as e:
        print(e)
        raise BentoMLException("Can't start or connect with clipper cluster")

    snapshot_path = generate_bentoml_deployment_snapshot_path(
        bento_service.name, bento_service.version, 'clipper')

    entry_py_content = DEFAULT_CLIPPER_ENTRY.format(api_name=api.name,
                                                    input_type=input_type)
    model_path = os.path.join(snapshot_path, 'bento')
    shutil.copytree(archive_path, model_path)

    with open(os.path.join(snapshot_path, 'clipper_entry.py'), 'w') as f:
        f.write(entry_py_content)

    with open(os.path.join(snapshot_path, 'Dockerfile-clipper'), 'w') as f:
        f.write(DOCKERFILE_CLIPPER)

    docker_api = docker.APIClient()
    image_tag = bento_service.name + '-clipper:' + bento_service.version
    for line in docker_api.build(path=snapshot_path, dockerfile='Dockerfile-clipper', tag=image_tag):
        process_docker_api_line(line)

    clipper_conn.deploy_model(name=application_name,
                              version=bento_service.version,
                              input_type=input_type, image=image_tag)

    return application_name
