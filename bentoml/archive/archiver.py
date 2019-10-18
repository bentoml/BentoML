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
import stat
import logging

from bentoml.exceptions import BentoMLException
from bentoml.archive.py_module_utils import copy_used_py_modules
from bentoml.archive.templates import (
    BENTO_MODEL_SETUP_PY_TEMPLATE,
    MANIFEST_IN_TEMPLATE,
    BENTO_SERVICE_DOCKERFILE_CPU_TEMPLATE,
    BENTO_SERVICE_DOCKERFILE_SAGEMAKER_TEMPLATE,
    INIT_PY_TEMPLATE,
    BENTO_INIT_SH_TEMPLATE,
)
from bentoml.archive.utils import add_local_bentoml_package_to_repo
from bentoml.utils import _is_bentoml_in_develop_mode
from bentoml.utils.usage_stats import track_save
from bentoml.archive.config import BentoArchiveConfig


DEFAULT_BENTO_ARCHIVE_DESCRIPTION = """\
# BentoML(bentoml.ai) generated model archive
"""

logger = logging.getLogger(__name__)


def _get_apis_list(bento_service):
    result = []
    for api in bento_service.get_service_apis():
        result.append(
            {
                "name": api.name,
                "handler_type": api.handler.__class__.__name__,
                "docs": api.doc,
            }
        )
    return result


def _get_artifacts_list(bento_service):
    result = []
    for artifact_name in bento_service.artifacts:
        artifact_spec = bento_service.artifacts[artifact_name].spec
        result.append(
            {'name': artifact_name, 'artifact_type': artifact_spec.__class__.__name__}
        )
    return result


def save_to_dir(bento_service, path, version=None):
    """Save given BentoService along with all its artifacts, source code and
    dependencies to target file path, assuming path exist and empty. If target path
    is not empty, this call may override existing files in the given path.

    Args:
        bento_service (bentoml.service.BentoService): a Bento Service instance
        path (str): Destination of where the bento service will be saved
    """
    track_save(bento_service)

    from bentoml.service import BentoService

    if not isinstance(bento_service, BentoService):
        raise BentoMLException(
            "save_to_dir only work with instance of custom BentoService class"
        )

    if version is not None:
        bento_service.set_version(version)

    if not os.path.exists(path):
        raise BentoMLException("Directory '{}' not found".format(path))

    for spec in bento_service._artifacts_spec:
        if spec.name not in bento_service.artifacts:
            logger.warning(
                "Missing declared artifact '%s' for BentoService '%s'",
                spec.name,
                bento_service.name,
            )

    module_base_path = os.path.join(path, bento_service.name)
    os.mkdir(module_base_path)

    # write README.md with user model's docstring
    if bento_service.__class__.__doc__:
        model_description = bento_service.__class__.__doc__.strip()
    else:
        model_description = DEFAULT_BENTO_ARCHIVE_DESCRIPTION
    with open(os.path.join(path, "README.md"), "w") as f:
        f.write(model_description)

    # save all model artifacts to 'base_path/name/artifacts/' directory
    if bento_service.artifacts:
        bento_service.artifacts.save(module_base_path)

    # write conda environment, requirement.txt
    bento_service.env.save(path)

    # TODO: add bentoml.find_packages helper for more fine grained control over this
    # process, e.g. packages=find_packages(base, [], exclude=[], used_module_only=True)
    # copy over all custom model code
    module_name, module_file = copy_used_py_modules(
        bento_service.__class__.__module__, os.path.join(path, bento_service.name)
    )

    # create __init__.py
    with open(os.path.join(path, bento_service.name, "__init__.py"), "w") as f:
        f.write(
            INIT_PY_TEMPLATE.format(
                service_name=bento_service.name,
                module_name=module_name,
                pypi_package_version=bento_service.version,
            )
        )

    # write setup.py, make exported model pip installable
    setup_py_content = BENTO_MODEL_SETUP_PY_TEMPLATE.format(
        name=bento_service.name,
        pypi_package_version=bento_service.version,
        long_description=model_description,
    )
    with open(os.path.join(path, "setup.py"), "w") as f:
        f.write(setup_py_content)

    with open(os.path.join(path, "MANIFEST.in"), "w") as f:
        f.write(MANIFEST_IN_TEMPLATE.format(service_name=bento_service.name))

    # write Dockerfile
    with open(os.path.join(path, "Dockerfile"), "w") as f:
        f.write(BENTO_SERVICE_DOCKERFILE_CPU_TEMPLATE)
    with open(os.path.join(path, "Dockerfile-sagemaker"), "w") as f:
        f.write(BENTO_SERVICE_DOCKERFILE_SAGEMAKER_TEMPLATE)

    # write bento init sh for install targz bundles
    bentoml_init_script_file = os.path.join(path, 'bentoml_init.sh')
    with open(bentoml_init_script_file, 'w') as f:
        f.write(BENTO_INIT_SH_TEMPLATE)
    # chmod +x bentoml_init_script file
    st = os.stat(bentoml_init_script_file)
    os.chmod(bentoml_init_script_file, st.st_mode | stat.S_IEXEC)

    # write bentoml.yml
    config = BentoArchiveConfig()
    config["metadata"].update(
        {
            "service_name": bento_service.name,
            "service_version": bento_service.version,
            "module_name": module_name,
            "module_file": module_file,
        }
    )
    config["env"] = bento_service.env.to_dict()
    config['apis'] = _get_apis_list(bento_service)
    config['artifacts'] = _get_artifacts_list(bento_service)

    config.write_to_path(path)
    # Also write bentoml.yml to module base path to make it accessible
    # as package data after pip installed as a python package
    config.write_to_path(module_base_path)

    # if bentoml package in editor mode(pip install -e), will include
    # that bentoml package to bento archive
    if _is_bentoml_in_develop_mode():
        add_local_bentoml_package_to_repo(path)

    logger.info(
        "Successfully saved Bento '%s:%s' to path: %s",
        bento_service.name,
        bento_service.version,
        path,
    )
