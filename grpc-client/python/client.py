from __future__ import annotations

import asyncio
import logging

import numpy as np

from bentoml.client import Client

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bentoml.client import GrpcClient


logger = logging.getLogger(__name__)


async def arun(client: GrpcClient):

    res = await client.async_classify(np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.async_classify':\n%s", res)
    res = await client.async_call("classify", np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.async_call':\n%s", res)


def run(client: GrpcClient):
    res = client.classify(np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.classify':\n%s", res)
    res = client.call("classify", np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.call(bentoml_api_name='classify')':\n%s", res)

    with client.service():
        res = client.Call(
            api_name="classify",
            ndarray={"float_values": [5.9, 3, 5.1, 1.8], "shape": [1, 4], "dtype": 1},
        )
        logger.info("Result from 'client.Call' in a context manager:\n%s", res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-rwa", "--run-with-async", action="store_true", default=False)
    parser.add_argument("--grpc", action="store_true", default=False)
    args = parser.parse_args()

    c: GrpcClient = Client.from_url("localhost:3000", grpc=args.grpc)

    if args.run_with_async:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(arun(c))
        finally:
            loop.close()
            assert loop.is_closed()
    else:
        run(c)
