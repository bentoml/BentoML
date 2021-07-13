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
import json
import logging
from typing import NamedTuple, Tuple, Iterator, Dict

import docker

from bentoml.exceptions import BentoMLException, MissingDependencyException

UNARY = 'UNARY'
SERVER_STREAMING = 'SERVER_STREAMING'
CLIENT_STREAMING = 'CLIENT_STREAMING'
BIDI_STREAMING = 'BIDI_STREAMING'
UNKNOWN = 'UNKNOWN'


logger = logging.getLogger(__name__)


def ensure_node_available_or_raise():
    from subprocess import CalledProcessError, check_output

    try:
        check_output(['node', '--version'])
    except CalledProcessError as error:
        raise BentoMLException(
            'Error executing node command: {}'.format(error.output.decode())
        )
    except FileNotFoundError:
        raise MissingDependencyException(
            'Node is required for Yatai web UI. Please visit '
            'www.nodejs.org for instructions'
        )


def parse_grpc_url(url):
    '''
    >>> parse_grpc_url("grpcs://yatai.com:43/query")
    ('grpcs', 'yatai.com:43/query')
    >>> parse_grpc_url("yatai.com:43/query")
    (None, 'yatai.com:43/query')
    '''
    from urllib3.util import parse_url

    parts = parse_url(url)
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
        raise RuntimeError('Unknown request_streaming or response_streaming')


class MethodName(NamedTuple):
    """
    Represents a gRPC method name
    Attributes:
        package: This is defined by `package foo.bar`,
        designation in the protocol buffer definition
        service: service name in protocol buffer
        definition (eg: service SearchService { ... })
        method: method name
    """

    package: str
    service: str
    method: str

    @property
    def fully_qualified_service(self):
        """return the service name prefixed with package"""
        return f"{self.package}.{self.service}" if self.package else self.service


def parse_method_name(method_name: str) -> Tuple[MethodName, bool]:
    '''
    Infers the grpc service and method name from the handler_call_details.
    e.g. /package.ServiceName/MethodName
    '''
    parts = method_name.split("/")
    if len(parts) < 3:
        return MethodName("", "", ""), False
    _, package_service, method = parts
    *packages, service = package_service.rsplit(".", maxsplit=1)
    package = packages[0] if packages else ""
    return MethodName(package, service, method), True


def docker_build_logs(resp: Iterator):
    """
    Stream build logs to stderr.

    Args:
        resp (:obj:`Iterator`):
            blocking generator from docker.api.build

    Raises:
        docker.errors.BuildErrors:
            When errors occurs during build process. Usually
            this comes when generated Dockerfile are incorrect.
    """
    output: str = ""
    try:
        while True:
            try:
                # output logs to stdout
                # https://docker-py.readthedocs.io/en/stable/user_guides/multiplex.html
                output = next(resp).decode('utf-8')
                json_output: Dict = json.loads(output.strip('\r\n'))
                # output to stderr when running in docker
                if 'stream' in json_output:
                    logger.debug(json_output['stream'])
            except StopIteration:
                break
            except ValueError:
                logger.debug(f"Errors while building image:\n{output}")
    except docker.errors.BuildError as e:
        print(f"Failed to build container :\n{e.msg}")
        for line in e.build_log:
            if 'stream' in line:
                logger.debug(line['stream'].strip())
