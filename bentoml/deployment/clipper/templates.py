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


DEFAULT_CLIPPER_ENTRY = """\
from __future__ import print_function

import rpc # this is clipper's rpc.py module
import os
import sys

from bentoml import load_service_api

IMPORT_ERROR_RETURN_CODE = 3

api = load_service_api('/container/bento', '{api_name}')

class BentoClipperContainer(rpc.ModelContainerBase):
    def __init__(self):
        self.input_type = '{input_type}'

    def predict_ints(self, inputs):
        preds = api.handle_clipper_ints(inputs)
        return [str(p) for p in preds]

    def predict_floats(self, inputs):
        preds = api.handle_clipper_floats(inputs)
        return [str(p) for p in preds]

    def predict_doubles(self, inputs):
        preds = api.handle_clipper_doubles(inputs)
        return [str(p) for p in preds]

    def predict_bytes(self, inputs):
        preds = api.handle_clipper_bytes(inputs)
        return [str(p) for p in preds]

    def predict_strings(self, inputs):
        preds = api.handle_clipper_strings(inputs)
        return [str(p) for p in preds]


if __name__ == "__main__":
    print("Starting Bento service Clipper Containter")
    rpc_service = rpc.RPCService()

    try:
        model = BentoClipperContainer()
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)

    rpc_service.start(model)
"""

DOCKERFILE_CLIPPER = """\
FROM clipper/python36-closure-container:0.4.1

# Install miniconda3 for python. Copied from
# https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN set -x \
    && apt-get update --fix-missing \
    && apt-get install -y wget bzip2 ca-certificates curl git libpq-dev build-essential \
                          libzmq3-dev libzmq5 libzmq5-dev redis-server libsodium18 \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc


# update conda and setup environment and pre-install common ML libraries to speed up docker build
RUN conda update conda -y \
      && conda install pip numpy scipy \
      && pip install six bentoml


# copy over model files
COPY . /container
WORKDIR /container

# update conda base env
RUN conda env update -n base -f /container/bento/environment.yml
RUN pip install -r /container/bento/requirements.txt

# Install packages for clipper
RUN pip install clipper_admin pyzmq==17.0.* prometheus_client==0.1.* pyyaml>=4.2b1 jsonschema==2.6.* redis==2.10.* psutil==5.4.* flask==0.12.2

# run user defined setup script
RUN if [ -f /container/bento/setup.sh ]; then /bin/bash -c /container/bento/setup.sh; fi

ENV CLIPPER_MODEL_NAME={model_name}
ENV CLIPPER_MODEL_VERSION={model_version}

# Stop running entry point from base image
ENTRYPOINT []
# Run BentoML bundle for clipper
CMD ["python", "/container/clipper_entry.py"]
"""  # noqa: E501
