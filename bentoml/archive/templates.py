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
    url="https://github.com/bentoml/BentoML",
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
FROM continuumio/miniconda3:4.7.12

ENTRYPOINT [ "/bin/bash", "-c" ]

EXPOSE 5000

RUN set -x \\
     && apt-get update \\
     && apt-get install --no-install-recommends --no-install-suggests -y libpq-dev build-essential \\
     && rm -rf /var/lib/apt/lists/*

# update conda, pre-install BentoML base dependencies
RUN conda update conda -y \\
      && conda install pip numpy scipy \\
      && pip install gunicorn six

# copy over model files
COPY . /bento
WORKDIR /bento

# update conda base env
RUN conda env update -n base -f /bento/environment.yml
RUN pip install -r /bento/requirements.txt

 # Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /bento/bentoml_init.sh ]; then /bin/bash -c /bento/bentoml_init.sh; fi

# run user defined setup script
RUN if [ -f /bento/setup.sh ]; then /bin/bash -c /bento/setup.sh; fi

# Run Gunicorn server with path to module.
CMD ["bentoml serve-gunicorn /bento"]
"""  # noqa: E501

BENTO_SERVICE_DOCKERFILE_SAGEMAKER_TEMPLATE = """\
FROM continuumio/miniconda3:4.7.12

EXPOSE 8080

RUN set -x \\
     && apt-get update \\
     && apt-get install --no-install-recommends --no-install-suggests -y libpq-dev build-essential\\
     && apt-get install -y nginx \\
     && rm -rf /var/lib/apt/lists/*

# update conda, pre-install BentoML base dependencies
RUN conda update conda -y \\
      && conda install pip numpy scipy \\
      && pip install gunicorn six gevent

# copy over model files
COPY . /opt/program
WORKDIR /opt/program

# update conda base env
RUN conda env update -n base -f /opt/program/environment.yml
RUN pip install -r /opt/program/requirements.txt

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /bento/bentoml_init.sh ]; then /bin/bash -c /bento/bentoml_init.sh; fi

# run user defined setup script
RUN if [ -f /opt/program/setup.sh ]; then /bin/bash -c /opt/program/setup.sh; fi

ENV PATH="/opt/program:${PATH}"
"""  # noqa: E501


INIT_PY_TEMPLATE = """\
import os
import sys
import logging

from bentoml import archive
from bentoml.cli import create_bento_service_cli
from bentoml.utils.log import configure_logging

# By default, ignore warnings when loading BentoService installed as PyPI distribution
# CLI will change back to default log level in config(info), and by adding --quiet or
# --verbose CLI option, user can change the CLI output behavior
configure_logging(logging.ERROR)

__VERSION__ = "{pypi_package_version}"

__module_path = os.path.abspath(os.path.dirname(__file__))

{service_name} = archive.load_bento_service_class(__module_path)

cli=create_bento_service_cli(__module_path)


def load():
    return archive.load(__module_path)


__all__ = ['__version__', '{service_name}', 'load']
"""


BENTO_INIT_SH_TEMPLATE = """\
#!/bin/bash

for filename in ./bundled_pip_dependencies/*.tar.gz; do
    [ -e "$filename" ] || continue
    pip install "$filename" --ignore-installed
done
"""
