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

import subprocess

import urllib3
from bentoml.exceptions import BentoMLException, MissingDependencyException

UNARY = "UNARY"
SERVER_STREAMING = "SERVER_STREAMING"
CLIENT_STREAMING = "CLIENT_STREAMING"
BIDI_STREAMING = "BIDI_STREAMING"
UNKNOWN = "UNKNOWN"


def ensure_node_available_or_raise():
    try:
        subprocess.check_output(["node", "--version"])
    except subprocess.CalledProcessError as error:
        raise BentoMLException(
            "Error executing node command: {}".format(error.output.decode())
        )
    except FileNotFoundError:
        raise MissingDependencyException(
            "Node is required for Yatai web UI. Please visit "
            "www.nodejs.org for instructions"
        )


def parse_grpc_url(url):
    """
    >>> parse_grpc_url("grpcs://yatai.com:43/query")
    ('grpcs', 'yatai.com:43/query')
    >>> parse_grpc_url("yatai.com:43/query")
    (None, 'yatai.com:43/query')
    """
    parts = urllib3.util.parse_url(url)
    return parts.scheme, url.replace(f"{parts.scheme}://", "", 1)


def wrap_interator_inc_counter(
    iterator, counter, grpc_type, grpc_service_name, grpc_method_name,
):
    for item in iterator:
        counter.labels(
            grpc_type=grpc_type,
            grpc_service=grpc_service_name,
            grpc_method=grpc_method_name,
        ).inc()
        yield item


# get method name for given RPC handler
def get_method_type(request_streaming, response_streaming) -> str:
    if not request_streaming and not response_streaming:
        return UNARY
    elif not request_streaming and response_streaming:
        return SERVER_STREAMING
    elif request_streaming and not response_streaming:
        return CLIENT_STREAMING
    elif request_streaming and response_streaming:
        return BIDI_STREAMING
    else:
        raise RuntimeError("Unknown request_streaming or response_streaming")


def parse_method_name(handler_call_details):
    """
    Infers the grpc service and method name from the handler_call_details.
    """

    # e.g. /package.ServiceName/MethodName
    parts = handler_call_details.method.split("/")
    if len(parts) < 3:
        return "", "", False

    grpc_service_name, grpc_method_name = parts[1:3]
    return grpc_service_name, grpc_method_name, True
