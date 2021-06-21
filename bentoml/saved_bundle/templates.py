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


BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE = """\
import setuptools
try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
    try:
        # for pip >= 20.0
        from pip._internal.network.session import PipSession
    except ModuleNotFoundError:
        # for pip >= 10, < 20.0
        from pip._internal.download import PipSession
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

try:
    raw = parse_requirements('requirements.txt', session=PipSession())

    # pip >= 20.1 changed ParsedRequirement attribute from `req` to `requirement`
    install_reqs = []
    for i in raw:
        try:
            install_reqs.append(str(i.requirement))
        except AttributeError:
            install_reqs.append(str(i.req))
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

MODEL_SERVER_DOCKERFILE_CPU = """\
FROM {docker_base_image}

# Configure PIP install arguments, e.g. --index-url, --trusted-url, --extra-index-url
ARG EXTRA_PIP_INSTALL_ARGS=
ENV EXTRA_PIP_INSTALL_ARGS $EXTRA_PIP_INSTALL_ARGS

ARG UID=1034
ARG GID=1034
RUN groupadd -g $GID -o bentoml && useradd -m -u $UID -g $GID -o -r bentoml

ARG BUNDLE_PATH=/home/bentoml/bundle
ENV BUNDLE_PATH=$BUNDLE_PATH
ENV BENTOML_HOME=/home/bentoml/

RUN mkdir $BUNDLE_PATH && chown bentoml:bentoml $BUNDLE_PATH -R
WORKDIR $BUNDLE_PATH

# copy over the init script; copy over entrypoint scripts
COPY --chown=bentoml:bentoml bentoml-init.sh docker-entrypoint.sh ./
RUN chmod +x ./bentoml-init.sh

# Copy docker-entrypoint.sh again, because setup.sh might not exist. This prevent COPY command from failing.
COPY --chown=bentoml:bentoml docker-entrypoint.sh setup.s[h] ./
RUN ./bentoml-init.sh custom_setup

COPY --chown=bentoml:bentoml docker-entrypoint.sh python_versio[n] ./
RUN ./bentoml-init.sh ensure_python

COPY --chown=bentoml:bentoml environment.yml ./
RUN ./bentoml-init.sh restore_conda_env

COPY --chown=bentoml:bentoml requirements.txt ./
RUN ./bentoml-init.sh install_pip_packages

COPY --chown=bentoml:bentoml docker-entrypoint.sh bundled_pip_dependencie[s]  ./bundled_pip_dependencies/
RUN rm ./bundled_pip_dependencies/docker-entrypoint.sh && ./bentoml-init.sh install_bundled_pip_packages

# copy over model files
COPY --chown=bentoml:bentoml . ./

# Default port for BentoML Service
EXPOSE 5000

USER bentoml
RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT [ "./docker-entrypoint.sh" ]
CMD ["bentoml", "serve-gunicorn", "./"]
"""  # noqa: E501

INIT_PY_TEMPLATE = """\
import os
import sys
import logging

from bentoml import saved_bundle, configure_logging
from bentoml.cli.bento_service import create_bento_service_cli

# By default, ignore warnings when loading BentoService installed as PyPI distribution
# CLI will change back to default log level in config(info), and by adding --quiet or
# --verbose CLI option, user can change the CLI output behavior
configure_logging(logging_level=logging.ERROR)

__VERSION__ = "{pypi_package_version}"

__module_path = os.path.abspath(os.path.dirname(__file__))

{service_name} = saved_bundle.load_bento_service_class(__module_path)

cli=create_bento_service_cli(__module_path)


def load():
    return saved_bundle.load_from_dir(__module_path)


__all__ = ['__version__', '{service_name}', 'load']
"""
