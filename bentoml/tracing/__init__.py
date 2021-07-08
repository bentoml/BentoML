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

from functools import lru_cache
import logging
from typing import Optional

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer


logger = logging.getLogger(__name__)


@inject
@lru_cache(maxsize=1)
def get_tracer(
    tracer_type=Provide[BentoMLContainer.config.tracing.type],
    zipkin_server_url: Optional[str] = Provide[
        BentoMLContainer.config.tracing.zipkin.url
    ],
    jaeger_server_address: Optional[str] = Provide[
        BentoMLContainer.config.tracing.jaeger.address
    ],
    jaeger_server_port: Optional[str] = Provide[
        BentoMLContainer.config.tracing.jaeger.port
    ],
):
    # isinstance check here allow trace to be used where the top-level entry point has
    # not yet implemented the wiring of BentoML config
    # TODO: remove this check after PR1543 https://github.com/bentoml/BentoML/pull/1543
    # if isinstance(tracer_type, Provider):
    # tracer_type = None
    # if isinstance(zipkin_server_url, Provider):
    # zipkin_server_url = None
    # if isinstance(jaeger_server_address, Provider):
    # jaeger_server_address = None
    # if isinstance(jaeger_server_port, Provider):
    # jaeger_server_port = None

    if tracer_type and tracer_type.lower() == 'zipkin' and zipkin_server_url:
        from bentoml.tracing.zipkin import get_zipkin_tracer

        logger.debug(
            "Initializing global zipkin tracer for collector endpoint: "
            f"{zipkin_server_url}"
        )
        return get_zipkin_tracer(zipkin_server_url)
    elif (
        tracer_type
        and tracer_type == 'jaeger'
        and jaeger_server_address
        and jaeger_server_port
    ):
        from bentoml.tracing.jaeger import get_jaeger_tracer

        logger.debug(
            "Initializing global jaeger tracer for opentracing server at "
            f"{jaeger_server_address}:{jaeger_server_port}"
        )
        return get_jaeger_tracer(jaeger_server_address, jaeger_server_port)
    else:
        from bentoml.tracing.noop import NoopTracer

        logger.debug("Tracing is disabled. Initializing no-op tracer")
        return NoopTracer()
