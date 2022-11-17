from __future__ import annotations

import typing as t
import asyncio
from typing import TYPE_CHECKING

from starlette.requests import Request
from multipart.multipart import parse_options_header
from starlette.responses import Response

from .base import from_spec as io_descriptor_from_spec
from .base import IODescriptor
from ..types import LazyType
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ...grpc.utils import import_generated_stubs
from ..service.openapi import SUCCESS_DESCRIPTION
from ..utils.formparser import populate_multipart_requests
from ..utils.formparser import concat_to_multipart_response
from ..service.openapi.specification import Schema
from ..service.openapi.specification import MediaType

if TYPE_CHECKING:
    from types import UnionType

    from google.protobuf import message as _message
    from typing_extensions import Self

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_pb2 as pb_v1alpha1

    from .base import StubVersion
    from .base import OpenAPIResponse
    from ..context import InferenceApiContext as Context

else:
    pb, _ = import_generated_stubs()
    pb_v1alpha1, _ = import_generated_stubs(version="v1alpha1")


class Multipart(
    IODescriptor[t.Dict[str, t.Any]],
    descriptor_id="bentoml.io.Multipart",
    proto_fields=("multipart",),
):
    """
    :obj:`Multipart` defines API specification for the inputs/outputs of a Service, where inputs/outputs
    of a Service can receive/send a **multipart** request/responses as specified in your API function signature.

    A sample service implementation:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       from typing import TYPE_CHECKING
       from typing import Any

       import bentoml
       from bentoml.io import NumpyNdarray
       from bentoml.io import Multipart
       from bentoml.io import JSON

       if TYPE_CHECKING:
           from numpy.typing import NDArray

       runner = bentoml.sklearn.get("sklearn_model_clf").to_runner()

       svc = bentoml.Service("iris-classifier", runners=[runner])

       input_spec = Multipart(arr=NumpyNdarray(), annotations=JSON())
       output_spec = Multipart(output=NumpyNdarray(), result=JSON())

       @svc.api(input=input_spec, output=output_spec)
       async def predict(
           arr: NDArray[Any], annotations: dict[str, Any]
       ) -> dict[str, NDArray[Any] | dict[str, Any]]:
           res = await runner.run(arr)
           return {"output": res, "result": annotations}

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

               % curl -X POST -H "Content-Type: multipart/form-data" \\
                      -F annotations=@test.json -F arr='[5,4,3,2]' \\
                      http://0.0.0.0:3000/predict

               # --b1d72c201a064ecd92a17a412eb9208e
               # Content-Disposition: form-data; name="output"
               # content-length: 1
               # content-type: application/json

               # 1
               # --b1d72c201a064ecd92a17a412eb9208e
               # Content-Disposition: form-data; name="result"
               # content-length: 13
               # content-type: application/json

               # {"foo":"bar"}
               # --b1d72c201a064ecd92a17a412eb9208e--

        .. tab-item:: Python

           .. note::

              The following code snippet uses `requests_toolbelt <https://github.com/requests/toolbelt>`_.
              Install with ``pip install requests-toolbelt``.

           .. code-block:: python
              :caption: `request.py`

              import requests

              from requests_toolbelt.multipart.encoder import MultipartEncoder

              m = MultipartEncoder(
                  fields={
                      "field0": "value",
                      "field1": "value",
                      "field2": ("filename", open("test.json", "rb"), "application/json"),
                  }
              )

              requests.post(
                  "http://0.0.0.0:3000/predict", data=m, headers={"Content-Type": m.content_type}
              )

    Args:
        inputs: Dictionary consisting keys as inputs definition for a Multipart
                request/response, values as IODescriptor supported by BentoML. Currently,
                :obj:`Multipart` supports :obj:`Image`, :obj:`NumpyNdarray`, :obj:`PandasDataFrame`, :obj:`PandasSeries`, :obj:`Text`,
                and :obj:`File`.

                Make sure to match the input parameters in function signatures in an API function to the keys defined
                under :obj:`Multipart`:

                .. code-block:: bash

                   +----------------------------------------------------------------+
                   |                                                                |
                   |   +--------------------------------------------------------+   |
                   |   |                                                        |   |
                   |   |    Multipart(arr=NumpyNdarray(), annotations=JSON())   |   |
                   |   |               |                       |                |   |
                   |   +---------------+-----------------------+----------------+   |
                   |                   |                       |                    |
                   |                   |                       |                    |
                   |                   |                       |                    |
                   |                   +-----+        +--------+                    |
                   |                         |        |                             |
                   |         +---------------v--------v---------+                   |
                   |         |  def predict(arr, annotations):  |                   |
                   |         +----------------------------------+                   |
                   |                                                                |
                   +----------------------------------------------------------------+

    Returns:
        :obj:`Multipart`: IO Descriptor that represents a Multipart request/response.
    """

    mime_type = "multipart/form-data"

    def __init__(self, **inputs: IODescriptor[t.Any]):
        if any(isinstance(descriptor, Multipart) for descriptor in inputs.values()):
            raise InvalidArgument(
                "Multipart IO can not contain nested Multipart IO descriptor"
            ) from None
        self._inputs = inputs

    def __repr__(self) -> str:
        return f"Multipart({','.join([f'{k}={v}' for k,v in zip(self._inputs, map(repr, self._inputs.values()))])})"

    def preprocess_sample(self, sample: dict[str, t.Any]) -> t.Any:
        raise NotImplementedError("'from_sample' is not supported for Multipart.")

    def input_type(
        self,
    ) -> dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]:
        res: dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]] = {}
        for k, v in self._inputs.items():
            inp_type = v.input_type()
            if isinstance(inp_type, dict):
                raise TypeError(
                    "A multipart descriptor cannot take a multi-valued I/O descriptor as input"
                ) from None
            res[k] = inp_type

        return res

    def to_spec(self) -> dict[str, t.Any]:
        return {
            "id": self.descriptor_id,
            "args": {
                argname: descriptor.to_spec()
                for argname, descriptor in self._inputs.items()
            },
        }

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> Self:
        if "args" not in spec:
            raise InvalidArgument(f"Missing args key in Multipart spec: {spec}")
        return Multipart(
            **{
                argname: io_descriptor_from_spec(spec)
                for argname, spec in spec["args"].items()
            }
        )

    def openapi_schema(self) -> Schema:
        return Schema(
            type="object",
            properties={args: io.openapi_schema() for args, io in self._inputs.items()},
        )

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self) -> t.Any:
        return {args: io.openapi_example() for args, io in self._inputs.items()}

    def openapi_request_body(self) -> dict[str, t.Any]:
        return {
            "content": {
                self.mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "required": True,
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def openapi_responses(self) -> OpenAPIResponse:
        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {
                self.mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    async def from_http_request(self, request: Request) -> dict[str, t.Any]:
        ctype, _ = parse_options_header(request.headers["content-type"])
        if ctype != b"multipart/form-data":
            raise BentoMLException(
                f"{self.__class__.__name__} only accepts `multipart/form-data` as Content-Type header, got {ctype} instead."
            ) from None

        form_values = await populate_multipart_requests(request)

        res = {}
        for field, descriptor in self._inputs.items():
            if field not in form_values:
                break
            res[field] = descriptor.from_http_request(form_values[field])
        else:  # NOTE: This is similar to goto, when there is no break.
            to_populate = zip(self._inputs.values(), form_values.values())
            reqs = await asyncio.gather(
                *tuple(io_.from_http_request(req) for io_, req in to_populate)
            )
            res = dict(zip(self._inputs, reqs))

        return res

    async def to_http_response(
        self, obj: dict[str, t.Any], ctx: Context | None = None
    ) -> Response:
        resps = await asyncio.gather(
            *tuple(
                io_.to_http_response(obj[key], ctx) for key, io_ in self._inputs.items()
            )
        )
        return await concat_to_multipart_response(dict(zip(self._inputs, resps)), ctx)

    def validate_input_mapping(self, field: t.MutableMapping[str, t.Any]) -> None:
        if len(set(field) - set(self._inputs)) != 0:
            raise InvalidArgument(
                f"'{self!r}' accepts the following keys: {set(self._inputs)}. Given {field.__class__.__qualname__} has invalid fields: {set(field) - set(self._inputs)}",
            ) from None

    def _register_unstructure_proto_fn(self):
        self._unstructure_proto_fn.register_iter_fns(
            (
                (lambda v: issubclass(v, bytes), self._unstructure_proto_bytes),
                (
                    lambda v: issubclass(v, pb.Multipart),
                    self._unstructure_proto_multipart,
                ),
                (
                    lambda v: issubclass(v, pb_v1alpha1.Multipart),
                    self._unstructure_proto_multipart,
                ),
            )
        )

    def _unstructure_proto_bytes(self, _: t.Any) -> t.Any:
        raise InvalidArgument(
            f"cannot use 'serialized_bytes' with {self.__class__.__name__}"
        ) from None

    async def _unstructure_proto_multipart(
        self, field: pb.Multipart | pb_v1alpha1.Multipart
    ) -> dict[str, t.Any]:
        from bentoml.grpc.utils import validate_proto_fields

        message = field.fields
        self.validate_input_mapping(message)
        to_populate = {self._inputs[k]: message[k] for k in self._inputs}
        reqs = await asyncio.gather(
            *tuple(
                descriptor.from_proto(
                    getattr(
                        part,
                        validate_proto_fields(
                            part.WhichOneof("representation"), descriptor
                        ),
                    )
                )
                for descriptor, part in to_populate.items()
            )
        )
        return dict(zip(self._inputs.keys(), reqs))

    def _register_structure_proto_fn(self):
        self._structure_proto_fn.register_predicate(
            lambda v: issubclass(v, dict), self._structure_proto_multipart
        )

    async def _structure_proto_multipart(
        self, obj: dict[str, t.Any], _version: StubVersion
    ) -> _message.Message:
        if _version == "v1":
            stub = pb
        elif _version == "v1alpha1":
            stub = pb_v1alpha1

        self.validate_input_mapping(obj)
        resps = await asyncio.gather(
            *tuple(
                io_.to_proto(data, _version=_version)
                for io_, data in zip(self._inputs.values(), obj.values())
            )
        )
        return stub.Multipart(
            fields=dict(
                zip(
                    obj,
                    [
                        # TODO: support multiple proto_fields
                        stub.Part(**{io_.proto_fields[0]: resp})
                        for io_, resp in zip(self._inputs.values(), resps)
                    ],
                )
            )
        )
