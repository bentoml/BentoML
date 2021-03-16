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

from contextlib import contextmanager

from dependency_injector.wiring import Provide, inject

from bentoml.configuration.containers import BentoMLContainer


@inject
@contextmanager
def trace(
    zipkin_api_url: str = Provide[BentoMLContainer.config.tracing.zipkin_api_url],
    opentracing_server_address: str = Provide[
        BentoMLContainer.config.tracing.opentracing_server_address
    ],
    opentracing_server_port: str = Provide[
        BentoMLContainer.config.tracing.opentracing_server_port
    ],
    **kwargs,
):
    """
    synchronous tracing function, will choose relevant tracer based on config
    """

    if zipkin_api_url:
        from bentoml.tracing.zipkin import trace as _trace

        kwargs['server_url'] = zipkin_api_url

    elif opentracing_server_address:
        from bentoml.tracing.opentrace import trace as _trace

        kwargs['server_url'] = opentracing_server_address
        kwargs['port'] = opentracing_server_port

    else:
        yield None
        return

    with _trace(**kwargs) as scope:
        yield scope
    return


@inject
@contextmanager
def async_trace(
    zipkin_api_url: str = Provide[BentoMLContainer.config.tracing.zipkin_api_url],
    opentracing_server_address: str = Provide[
        BentoMLContainer.config.tracing.opentracing_server_address
    ],
    opentracing_server_port: str = Provide[
        BentoMLContainer.config.tracing.opentracing_server_port
    ],
    **kwargs,
):
    """
    asynchronous tracing function, will choose relevant tracer based on config
    """
    if zipkin_api_url:
        from bentoml.tracing.zipkin import async_trace as _async_trace

        kwargs['server_url'] = zipkin_api_url

    elif opentracing_server_address:
        from bentoml.tracing.opentrace import async_trace as _async_trace

        kwargs['server_url'] = opentracing_server_address
        kwargs['port'] = opentracing_server_port

    else:
        yield None
        return

    with _async_trace(**kwargs) as scope:
        yield scope
    return
