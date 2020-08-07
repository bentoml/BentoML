# Copyright 2020 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import NamedTuple, Union, Dict, List

from multidict import CIMultiDict

from bentoml.utils import cached_property


class HTTPRequest(NamedTuple):
    '''
    headers: tuple of key value pairs in bytes
    data: str
    '''

    headers: tuple
    data: str

    @cached_property
    def parsed_headers(self):
        return CIMultiDict(
            (hk.decode("latin1").lower(), hv.decode("latin1"))
            for hk, hv in self.headers or tuple()
        )

    @classmethod
    def from_flask_request(cls, request):
        # For non latin1 characters: https://tools.ietf.org/html/rfc8187
        # Also https://github.com/benoitc/gunicorn/issues/1778
        return cls(
            tuple((k.encode("latin1"), v.encode("latin1")) for k, v in request.headers),
            request.get_data(),
        )


class HTTPResponse(NamedTuple):
    status: int
    headers: tuple
    data: str

    def to_flask_response(self):
        import flask

        return flask.Response(
            headers=self.headers, response=self.data, status=self.status
        )


# https://tools.ietf.org/html/rfc7159#section-3
JsonSerializable = Union[bool, None, Dict, List, int, float, str]

# https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html
AwsLambdaEvent = Union[Dict, List, str, int, float, None]


class InferenceTask(NamedTuple):
    context: dict
    data: object


class InferenceResult(NamedTuple):
    context: dict
    data: object


class JsonInferenceTask(InferenceTask):
    data: str  # json string
