# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import tempfile
import uuid
from datetime import datetime

from bentoml.utils import Path
from bentoml.utils.py_module_utils import copy_used_py_modules
from bentoml.utils.exceptions import BentoMLException
from bentoml.utils.s3 import is_s3_url, upload_to_s3, download_from_s3
from bentoml.loader import load_bentoml_config
from bentoml.version import __version__ as BENTOML_VERSION

BENTO_MODEL_SETUP_PY_TEMPLATE = """\
import os
import pip
import logging
import pkg_resources
import setuptools

def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]

try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setuptools.setup(
    name='{name}',
    version='{pypi_package_version}',
    description="BentoML generated model module",
    long_description=\"\"\"{long_description}\"\"\",
    long_description_content_type="text/markdown",
    url="https://github.com/atalaya-io/BentoML",
    packages=setuptools.find_packages(),
    install_requires=install_reqs,
    include_package_data=True,
    package_data={{
        '{name}': ['bentoml.yml', 'artifacts/*']
    }},
    # TODO: add console entry_point for using model from cli
    # entry_points={{
    #
    # }}
)
"""

MANIFEST_IN_TEMPLATE = """\
include {service_name}/bentoml.yml
graft {service_name}/artifacts
"""

# TODO: add setup.sh and run
BENTO_SERVER_SINGLE_MODEL_DOCKERFILE_TEMPLATE = """\
FROM continuumio/miniconda3

ENTRYPOINT [ "/bin/bash", "-c" ]

EXPOSE 5000

RUN apt-get update && apt-get install -y \
      libpq-dev \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

# copy over model files
COPY . /bento
WORKDIR /bento

ARG conda_env={conda_env_name}

# update conda and setup environment
RUN conda update conda -y

# create conda env
RUN conda env create -f /bento/environment.yml

ENV PATH /opt/conda/envs/$conda_env/bin:$PATH

RUN conda install pip && pip install -r /bento/requirements.txt

# Run bento server with path to bento archive
CMD ["bentoml serve --model-path=/bento"]
"""

# TODO: improve this with import hooks PEP302?
INIT_PY_TEMPLATE = """\
import os
import sys

__VERSION__ = "{pypi_package_version}"

__module_path = os.path.abspath(os.path.split(__file__)[0])

# Prepend __module_path to sys.path, to avoid name conflicts with other installed modules
sys.path.insert(0, __module_path)
import {module_name}
sys.path.remove(__module_path)

# Set _bento_module_path, which tells the model where to load its artifacts
{service_name} = {module_name}.{service_name}
{service_name}._bento_module_path = __module_path

__all__ = ['__version__', '{service_name}']
"""

BENTOML_CONFIG_YAML_TEMPLATE = """\
service_name: {service_name}
service_version: {service_version}
module_name: {module_name}
module_file: {module_file}
created_at: {created_at}
bentoml_version: {bentoml_version}
"""

DEFAULT_BENTO_ARCHIVE_DESCRIPTION = """\
# BentoML(bentoml.ai) generated model archive
"""


def _validate_version_str(version_str):
    """
    Validate that version str starts with char, has length >= 4
    """
    regex = r"^[a-zA-Z][a-zA-Z0-9_]{3,}"
    if re.match(regex, version_str) is None:
        raise ValueError('Invalid version format: "{}"'.format(version_str))


def _generate_new_version_str():
    """
    Generate a version string in the format of YYYY_MM_DD_RandomHash
    """
    time_obj = datetime.now()
    date_string = time_obj.strftime('%Y_%m_%d')
    random_hash = uuid.uuid4().hex[:8]

    return 'b' + date_string + '_' + random_hash


# TODO: make pypi version consistent with archive version, and use Semantic versioning for both
def save(bento_service, dst, version=None, pypi_package_version="1.0.0"):
    """
    Save given BentoService along with all artifacts to target path
    """

    if version is None:
        version = _generate_new_version_str()
    _validate_version_str(version)

    s3_url = None
    if is_s3_url(dst):
        s3_url = os.path.join(dst, bento_service.name, version)
        # TODO: check s3_url not exist, otherwise raise exception
        temp_dir = tempfile.mkdtemp()
        Path(temp_dir, bento_service.name).mkdir(parents=True, exist_ok=True)
        # Update path to subfolder in the form of 'base/service_name/version/'
        path = os.path.join(temp_dir, bento_service.name, version)
    else:
        Path(os.path.join(dst), bento_service.name).mkdir(parents=True, exist_ok=True)
        # Update path to subfolder in the form of 'base/service_name/version/'
        path = os.path.join(dst, bento_service.name, version)

        if os.path.exists(path):
            raise ValueError("Version {version} in Path: {dst} already "
                             "exist.".format(version=version, dst=dst))

    os.mkdir(path)
    module_base_path = os.path.join(path, bento_service.name)
    os.mkdir(module_base_path)

    # write README.md with user model's docstring
    if bento_service.__class__.__doc__:
        model_description = bento_service.__class__.__doc__.strip()
    else:
        model_description = DEFAULT_BENTO_ARCHIVE_DESCRIPTION
    with open(os.path.join(path, 'README.md'), 'w') as f:
        f.write(model_description)

    # save all model artifacts to 'base_path/name/artifacts/' directory
    bento_service.artifacts.save(module_base_path)

    # write conda environment, requirement.txt
    bento_service.env.save(path)

    # TODO: add bentoml.find_packages helper for more fine grained control over
    # this process, e.g. packages=find_packages(base, [], exclude=[], used_module_only=True)
    # copy over all custom model code
    module_name, module_file = copy_used_py_modules(bento_service.__class__.__module__,
                                                    os.path.join(path, bento_service.name))

    if os.path.isabs(module_file):
        module_file = module_name.replace('.', os.sep) + '.py'

    # create __init__.py
    with open(os.path.join(path, bento_service.name, '__init__.py'), "w") as f:
        f.write(
            INIT_PY_TEMPLATE.format(service_name=bento_service.name, module_name=module_name,
                                    pypi_package_version=pypi_package_version))

    # write setup.py, make exported model pip installable
    setup_py_content = BENTO_MODEL_SETUP_PY_TEMPLATE.format(
        name=bento_service.name, pypi_package_version=pypi_package_version,
        long_description=model_description)
    with open(os.path.join(path, 'setup.py'), 'w') as f:
        f.write(setup_py_content)

    with open(os.path.join(path, 'MANIFEST.in'), 'w') as f:
        f.write(MANIFEST_IN_TEMPLATE.format(service_name=bento_service.name))

    # write Dockerfile
    with open(os.path.join(path, 'Dockerfile'), 'w') as f:
        f.write(
            BENTO_SERVER_SINGLE_MODEL_DOCKERFILE_TEMPLATE.format(
                conda_env_name=bento_service.env.get_conda_env_name()))

    # write bentoml.yml
    bentoml_yml_content = BENTOML_CONFIG_YAML_TEMPLATE.format(
        service_name=bento_service.name, bentoml_version=BENTOML_VERSION, service_version=version,
        module_name=module_name, module_file=module_file, created_at=str(datetime.now()))
    with open(os.path.join(path, 'bentoml.yml'), 'w') as f:
        f.write(bentoml_yml_content)
    # Also write bentoml.yml to module base path to make it accessible
    # as package data after pip installed as a python package
    with open(os.path.join(module_base_path, 'bentoml.yml'), 'w') as f:
        f.write(bentoml_yml_content)

    if s3_url:
        upload_to_s3(s3_url, path)
        return s3_url
    else:
        return path


# TODO: consolidate this with loader module
def load(bento_service_cls, path=None):
    # TODO: add model.env.verify() to check dependencies and python version etc

    if bento_service_cls._bento_module_path is not None:
        # When calling load from pip installled bento model, use installed
        # python package for loading and the same path for '/artifacts'

        # TODO: warn user that 'path' parameter is ignored if it's not None here
        path = bento_service_cls._bento_module_path
        artifacts_path = path
    else:
        if path is None:
            raise BentoMLException("Loading path is required for BentoArchive: {}.".format(
                bento_service_cls.name()))

        # When calling load on generated archive directory, look for /artifacts
        # directory under module sub-directory
        if is_s3_url(path):
            temporary_path = tempfile.mkdtemp()
            download_from_s3(path, temporary_path)
            # Use loacl temp path for the following loading operations
            path = temporary_path

        artifacts_path = os.path.join(path, bento_service_cls.name())

    bentoml_config = load_bentoml_config(path)

    bento_service = bento_service_cls.load(artifacts_path)

    bento_service._version = bentoml_config['service_version']
    return bento_service
