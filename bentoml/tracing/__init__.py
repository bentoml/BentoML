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

import logging
from functools import lru_cache

from dependency_injector.wiring import Provide, inject

from bentoml.configuration.containers import BentoMLContainer


logger = logging.getLogger(__name__)

_tracer = None


@inject
@lru_cache(maxsize=1)
def get_tracer(
    type: str = Provide[BentoMLContainer.config.tracing.type],
    zipkin_server_url: str = Provide[BentoMLContainer.config.tracing.zipkin.url],
    opentracing_server_address: str = Provide[
        BentoMLContainer.config.tracing.opentracing.address
    ],
    opentracing_server_port: str = Provide[
        BentoMLContainer.config.tracing.opentracing.port
    ],
):
    print("Get_Tracer called", type, zipkin_server_url)
    # isinstance check here allow trace to be used where the top-level entry point has
    # not yet implemented the wiring of BentoML config
    # TODO: remove this check after PR1543 https://github.com/bentoml/BentoML/pull/1543
    if isinstance(type, Provide):
        type = None
    if isinstance(zipkin_server_url, Provide):
        zipkin_server_url = None
    if isinstance(opentracing_server_address, Provide):
        opentracing_server_address = None
    if isinstance(opentracing_server_port, Provide):
        opentracing_server_port = None

    if type and type.lower() == 'zipkin' and zipkin_server_url:
        from bentoml.tracing.zipkin import get_zipkin_tracer

        logger.info(
            "Initializing global zipkin tracer for collector endpoint: "
            f"{zipkin_server_url}"
        )
        return get_zipkin_tracer(zipkin_server_url)

    if (
        type
        and type.lower() == 'opentracing'
        and opentracing_server_address
        and opentracing_server_port
    ):
        from bentoml.tracing.opentrace import get_opentracing_tracer

        logger.info(
            "Initializing global jaeger tracer for opentracing server at "
            f"{opentracing_server_address}:{opentracing_server_port}"
        )
        return get_opentracing_tracer(
            opentracing_server_address, opentracing_server_port
        )

    from bentoml.tracing.noop import NoopTracer

    return NoopTracer()
