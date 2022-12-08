from __future__ import annotations

import asyncio
import logging

import numpy as np

import bentoml


async def async_run(client: bentoml.client.Client):

    res = await client.async_classify(np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.async_classify':\n%s", res)
    res = await client.async_call("classify", np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.async_call':\n%s", res)


def run(client: bentoml.client.Client):
    res = client.classify(np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.classify':\n%s", res)
    res = client.call("classify", np.array([[5.9, 3, 5.1, 1.8]]))
    logger.info("Result from 'client.call(bentoml_api_name='classify')':\n%s", res)


if __name__ == "__main__":
    import argparse

    logger = logging.getLogger(__name__)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sync", action="store_true", default=False)
    args = parser.parse_args()

    c = bentoml.client.Client.from_url("localhost:3000")

    if args.sync:
        run(c)
    else:
        asyncio.run(async_run(c))
