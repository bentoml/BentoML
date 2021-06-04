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
FROM python:{python_version}

COPY requirements.txt /
RUN pip install -r ./requirements.txt --no-cache-dir

COPY . /

ENV PORT 5000
EXPOSE $PORT

CMD ["bentoml", "serve", "."]

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

ENTHIRE_BACKEND_DOCKERFILE_TEMPLATE="""\
FROM python:3.8-slim

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

RUN apt-get update
RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY BentoML-0.12.1+26.g404685a.dirty-py3-none-any.whl .
RUN pip install BentoML-0.12.1+26.g404685a.dirty-py3-none-any.whl


COPY . .

# the env var $PORT is required by heroku container runtime
EXPOSE 8080

USER bentoml
CMD ["bentoml", "serve", "./"]
"""

ENTHIRE_FRONTEND_DOCKERFILE_TEMPLATE="""\
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
"""

ENTHIRE_DOCKER_COMPOSE="""\
version: '3'

services:
  frontend:
    build: frontend
    ports:
      - 8501:8501
    depends_on:
      - backend
    volumes:
        - ./storage:/storage
  backend:
    build: backend
    ports:
      - 8080:8080
    volumes:
      - ./storage:/storage
"""

ENTHIRE_STREAMLIT_TEMPLATE = """\
import requests
import streamlit as st

def return_input_type(input_selection=None, param=None):
    if input_selection=="text_input":return st.text_input(f'Write some text for {param}'),
    if input_selection=="num_input": return st.number_input(f'Enter a number for {param}'),
    if input_selection=="text_area": return st.text_area(f'Area for textual entry for {param}'),
    if input_selection=="date_input":return  st.date_input(f'Date input for {param}'),
    if input_selection=="time_input":return  st.time_input(f'Time entry for {param}'),
    if input_selection=="file_uploader":return  st.file_uploader(f'File uploader for {param}')

def get_keys():
    return ['text_input', 'num_input', 'text_area', 'date_input', 'time_input', 'file_uploader']

def selection(param_title):
    return st.selectbox(f'Input Type for -> {param_title}', options=get_keys())

st.title("Your MODEL PLAYSTORE")

st.sidebar.header("Your APIs")\n
"""
