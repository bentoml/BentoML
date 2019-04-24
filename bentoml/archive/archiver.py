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
import uuid
from datetime import datetime

from bentoml.utils import Path
from bentoml.utils.s3 import is_s3_url, upload_to_s3
from bentoml.utils.tempdir import TempDirectory
from bentoml.version import __version__ as BENTOML_VERSION
from bentoml.archive.py_module_utils import copy_used_py_modules

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
    entry_points={{
        'console_scripts': [
            '{name}={name}:cli',
        ],
    }}
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

RUN conda install pip && pip install -r /bento/requirements.txt && pip install gunicorn

# Run Gunicorn server with path to module.
CMD ["bentoml serve-gunicorn /bento"]
"""

# TODO: improve this with import hooks PEP302?
INIT_PY_TEMPLATE = """\
import os
import sys

from bentoml import archive
from bentoml.cli import create_bentoml_cli

__VERSION__ = "{pypi_package_version}"

__module_path = os.path.abspath(os.path.dirname(__file__))

{service_name} = archive.load_bento_service_class(__module_path)

cli=create_bentoml_cli(__module_path)


def load():
    return archive.load(__module_path)


__all__ = ['__version__', '{service_name}', 'load']
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
    Validate that version str format:
    * Consist of only ALPHA / DIGIT / "-" / "." / "_"
    * Length between 1-128
    """
    regex = r"[A-Za-z0-9_.-]{1,128}\Z"
    if re.match(regex, version_str) is None:
        raise ValueError('Invalid BentoArchive version: "{}", it can only consist'
                         ' ALPHA / DIGIT / "-" / "." / "_", and must be less than'
                         '128 characthers'.format(version_str))


def _generate_new_version_str():
    """
    Generate a version string in the format of YYYY-MM-DD-Hash
    """
    time_obj = datetime.now()
    date_string = time_obj.strftime('%Y_%m_%d')
    random_hash = uuid.uuid4().hex[:8]

    return date_string + '_' + random_hash


def save(bento_service, dst, version=None):
    """
    Save given BentoService along with all artifacts to target path
    """

    if version is None:
        version = _generate_new_version_str()
    _validate_version_str(version)

    if bento_service._version_major is not None and bento_service._version_minor is not None:
        # BentoML uses semantic versioning for BentoService distribution
        # when user specified the MAJOR and MINOR version number along with
        # the BentoService class definition with '@ver' decorator.
        # The parameter version(or auto generated version) here will be used as
        # PATCH field in the final version:
        version = '.'.join(
            [str(bento_service._version_major),
             str(bento_service._version_minor), version])

    # Full path containing saved BentoArchive, it the dst path with service name
    # and service version as prefix. e.g.:
    # - s3://my-bucket/base_path => s3://my-bucket/base_path/service_name/version/
    # - /tmp/my_bento_archive/ => /tmp/my_bento_archive/service_name/version/
    full_saved_path = os.path.join(dst, bento_service.name, version)

    if is_s3_url(dst):
        with TempDirectory() as tempdir:
            _save(bento_service, tempdir, version)
            upload_to_s3(full_saved_path, tempdir)
    else:
        _save(bento_service, dst, version)

    return full_saved_path


def _save(bento_service, dst, version=None):
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

    # create __init__.py
    with open(os.path.join(path, bento_service.name, '__init__.py'), "w") as f:
        f.write(
            INIT_PY_TEMPLATE.format(service_name=bento_service.name, module_name=module_name,
                                    pypi_package_version=version))

    # write setup.py, make exported model pip installable
    setup_py_content = BENTO_MODEL_SETUP_PY_TEMPLATE.format(
        name=bento_service.name, pypi_package_version=version, long_description=model_description)
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
