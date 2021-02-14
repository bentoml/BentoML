# Copyright 2021 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# List of APIs for accessing remote or local yatai service via Python

from contextlib import contextmanager

from bentoml import config


ZIPKIN_API_URL = config("tracing").get("zipkin_api_url")
OPENTRACING_SERVER_ADDRESS = config("tracing").get("opentracing_server_address")
OPENTRACING_SERVER_PORT = config("tracing").get("opentracing_server_port")


@contextmanager
def trace(*args, **kwargs):
    """
    synchronous tracing function, will choose relevant tracer
    """

    if ZIPKIN_API_URL:
        from bentoml.tracing.trace import trace as _trace

        new_args = [ZIPKIN_API_URL, *args]

    elif OPENTRACING_SERVER_ADDRESS:
        from bentoml.tracing.opentrace import trace as _trace

        new_args = [OPENTRACING_SERVER_ADDRESS, *args]
        kwargs['port'] = OPENTRACING_SERVER_PORT

    else:
        yield
        return

    with _trace(*new_args, **kwargs) as scope:
        yield scope
    return


@contextmanager
def async_trace(*args, **kwargs):
    """
    asynchronous tracing function, will choose relevant tracer
    """
    if ZIPKIN_API_URL:
        from bentoml.tracing.trace import async_trace as _async_trace

        new_args = [ZIPKIN_API_URL, *args]

    elif OPENTRACING_SERVER_ADDRESS:
        from bentoml.tracing.opentrace import async_trace as _async_trace

        new_args = [OPENTRACING_SERVER_ADDRESS, *args]
        kwargs['port'] = OPENTRACING_SERVER_PORT

    else:
        yield
        return

    with _async_trace(*new_args, **kwargs) as scope:
        yield scope
    return
