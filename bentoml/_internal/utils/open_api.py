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

from collections import OrderedDict

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer


@inject
def get_open_api_spec_json(
    bento_service,
    enable_metrics: bool = Provide[
        BentoMLContainer.config.bento_server.metrics.enabled
    ],
    enable_feedback: bool = Provide[
        BentoMLContainer.config.bento_server.feedback.enabled
    ],
):
    """
    The docs for all endpoints in Open API format.
    """
    docs = OrderedDict(
        openapi="3.0.0",
        info=OrderedDict(
            version=bento_service.version,
            title=bento_service.name,
            description="To get a client SDK, copy all content from <a "
            "href=\"/docs.json\">docs</a> and paste into "
            "<a href=\"https://editor.swagger.io\">editor.swagger.io</a> then click "
            "the tab <strong>Generate Client</strong> and choose the language.",
        ),
        tags=[{"name": "infra"}, {"name": "app"}],
    )

    paths = OrderedDict()
    default_response = {"200": {"description": "success"}}

    paths["/healthz"] = OrderedDict(
        get=OrderedDict(
            tags=["infra"],
            description="Health check endpoint. Expecting an empty response with status"
            " code 200 when the service is in health state",
            responses=default_response,
        )
    )

    paths["/metadata"] = OrderedDict(
        get=OrderedDict(
            tags=["infra"],
            description="BentoService metadata endpoint. Returns the service's"
            "`bentoml.yml` in JSON format.",
            responses=default_response,
        )
    )

    if enable_metrics:
        paths["/metrics"] = OrderedDict(
            get=OrderedDict(
                tags=["infra"],
                description="Prometheus metrics endpoint",
                responses=default_response,
            )
        )
    if enable_feedback:
        paths["/feedback"] = OrderedDict(
            post=OrderedDict(
                tags=["infra"],
                description="Provide feedback to prediction results from BentoService. "
                "Expecting feedback request payload in JSON format "
                "and requires `request_id` field, which can be obtained "
                "from any BentoService prediction response's header. "
                "Only last key will be considered if keys are repeated.",
                requestBody=OrderedDict(
                    required=True,
                    content={
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["request_id"],
                                "properties": {"request_id": {"type": "uuid"}},
                            },
                            "example": {
                                "request_id": "cf420b0f-15fa-013d-a37b-12345678c321",
                                "example_feedback": "key-value pair can be anything",
                            },
                        }
                    },
                ),
                responses=default_response,
            ),
        )

    for api in bento_service.inference_apis:
        path = "/{}".format(api.route)
        paths[path] = OrderedDict(
            post=OrderedDict(
                tags=["app"],
                description=api.doc,
                requestBody=OrderedDict(required=True, content=api.request_schema),
                responses=default_response,
            )
        )

    docs["paths"] = paths
    return docs
