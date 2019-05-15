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

BENTO_SERVICE_DOCKERFILE_CPU_TEMPLATE = """\
FROM continuumio/miniconda3

ENTRYPOINT [ "/bin/bash", "-c" ]

EXPOSE 5000

RUN set -x \
     && apt-get update \
     && apt-get install --no-install-recommends --no-install-suggests -y libpq-dev build-essential \
     && rm -rf /var/lib/apt/lists/*

# update conda and setup environment and pre-install common ML libraries to speed up docker build
RUN conda update conda -y \
      && conda install pip numpy scipy \
      && pip install gunicorn six

# copy over model files
COPY . /bento
WORKDIR /bento

# update conda base env
RUN conda env update -n base -f /bento/environment.yml
RUN pip install -r /bento/requirements.txt

# run user defined setup script
RUN if [ -f /bento/setup.sh ]; then /bin/bash -c /bento/setup.sh; fi

# Run Gunicorn server with path to module.
CMD ["bentoml serve-gunicorn /bento"]
"""

BENTO_SERVICE_DOCKERFILE_SAGEMAKER_TEMPLATE = """\
FROM continuumio/miniconda3

EXPOSE 8080

RUN set -x \
     && apt-get update \
     && apt-get install --no-install-recommends --no-install-suggests -y libpq-dev build-essential\
     && apt-get install -y nginx \
     && rm -rf /var/lib/apt/lists/*

# update conda and setup environment and pre-install common ML libraries to speed up docker build
RUN conda update conda -y \
      && conda install pip numpy scipy \
      && pip install gunicorn six gevent

# copy over model files
COPY . /opt/program
WORKDIR /opt/program

# update conda base env
RUN conda env update -n base -f /opt/program/environment.yml
RUN pip install -r /opt/program/requirements.txt

# run user defined setup script
RUN if [ -f /opt/program/setup.sh ]; then /bin/bash -c /opt/program/setup.sh; fi

ENV PATH="/opt/program:${PATH}"
"""

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
