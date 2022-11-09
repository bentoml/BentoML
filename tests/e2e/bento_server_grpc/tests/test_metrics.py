from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bentoml.grpc.utils import import_generated_stubs
from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import async_client_call
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")
    pb, _ = import_generated_stubs()


@pytest.mark.asyncio
async def test_metrics_available(host: str, img_file: str):
    with open(str(img_file), "rb") as f:
        fb = f.read()

    async with create_channel(host) as channel:
        await async_client_call(
            "predict_multi_images",
            channel=channel,
            data={
                "multipart": {
                    "fields": {
                        "original": pb.Part(
                            file=pb.File(kind=pb.File.FILE_TYPE_BMP, content=fb)
                        ),
                        "compared": pb.Part(
                            file=pb.File(kind=pb.File.FILE_TYPE_BMP, content=fb)
                        ),
                    }
                }
            },
        )
        await async_client_call(
            "ensure_metrics_are_registered",
            channel=channel,
            data={"text": wrappers_pb2.StringValue(value="input_string")},
        )
