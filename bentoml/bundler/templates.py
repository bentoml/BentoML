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


BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE = """\
import setuptools
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

try:
    raw = parse_requirements('requirements.txt', session=PipSession())
    install_reqs =  [str(i.req) for i in raw]
except Exception:
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

# pre-install BentoML base dependencies
RUN conda install pip numpy scipy \\
      && pip install gunicorn

# copy over model files
COPY . /bento
WORKDIR /bento

# run user defined setup script
RUN if [ -f /bento/setup.sh ]; then /bin/bash -c /bento/setup.sh; fi

# update conda base env
RUN conda env update -n base -f /bento/environment.yml
RUN pip install -r /bento/requirements.txt

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /bento/bentoml_init.sh ]; then /bin/bash -c /bento/bentoml_init.sh; fi

# Run Gunicorn server with path to module.
CMD ["bentoml serve-gunicorn /bento"]
"""  # noqa: E501


INIT_PY_TEMPLATE = """\
import os
import sys
import logging

from bentoml import bundler
from bentoml.cli import create_bento_service_cli
from bentoml.utils.log import configure_logging

# By default, ignore warnings when loading BentoService installed as PyPI distribution
# CLI will change back to default log level in config(info), and by adding --quiet or
# --verbose CLI option, user can change the CLI output behavior
configure_logging(logging.ERROR)

__VERSION__ = "{pypi_package_version}"

__module_path = os.path.abspath(os.path.dirname(__file__))

{service_name} = bundler.load_bento_service_class(__module_path)

cli=create_bento_service_cli(__module_path)


def load():
    return bundler.load(__module_path)


__all__ = ['__version__', '{service_name}', 'load']
"""


BENTO_INIT_SH_TEMPLATE = """\
#!/bin/bash

for filename in ./bundled_pip_dependencies/*.tar.gz; do
    [ -e "$filename" ] || continue
    pip install -U "$filename"
done
"""
