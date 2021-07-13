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


from bentoml.adapters.base_input import BaseInputAdapter

ADAPTER_TYPE_TO_INPUT_TYPE = {
    "ClipperBytesInput": "bytes",
    "ClipperIntsInput": "ints",
    "ClipperFloatsInput": "floats",
    "ClipperDoublesInput": "doubles",
    "ClipperStringsInput": "strings",
}


class ClipperInput(BaseInputAdapter):
    """
    A special input adapter that should only be used when deploying BentoService
     with Clipper(http://clipper.ai/)

    ClipperInput are not regular InputAdapter, they don't work with REST
    API server nor BentoML CLI.
    """

    @property
    def pip_dependencies(self):
        # 'clipper_admin' package is only required on the machine deploying BentoService
        # to a clipper cluster, not required by the API Server itself
        return []

    def from_http_request(self, req):
        raise NotImplementedError(
            "ClipperInput does not support handling REST API prediction request"
        )

    def from_cli(self, cli_args):
        raise NotImplementedError(
            "ClipperInput is not supported to be used with BentoML CLI"
        )

    def from_inference_job(self, *args, **kwargs):
        raise NotImplementedError(
            "" "ClipperInput is not supported to be used with BentoML function calling"
        )

    def from_aws_lambda_event(self, event):
        raise NotImplementedError(
            "ClipperInput is not supported in AWS Lambda Deployment"
        )


# pylint: disable=abstract-method
class ClipperBytesInput(ClipperInput):
    """
    ClipperInput that deals with input type Bytes
    """


class ClipperFloatsInput(ClipperInput):
    """
    ClipperInput that deals with input type Floats
    """


class ClipperIntsInput(ClipperInput):
    """
    ClipperInput that deals with input type Ints
    """


class ClipperDoublesInput(ClipperInput):
    """
    ClipperInput that deals with input type Doubles
    """


class ClipperStringsInput(ClipperInput):
    """
    ClipperInput that deals with input type String
    """
