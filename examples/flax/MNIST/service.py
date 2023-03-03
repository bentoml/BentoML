from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import jax.numpy as jnp
from PIL.Image import Image as PILImage

import bentoml

if TYPE_CHECKING:
    from numpy.typing import NDArray

mnist_runner = bentoml.flax.get("mnist_flax").to_runner()

svc = bentoml.Service(name="mnist_flax", runners=[mnist_runner])


@svc.api(input=bentoml.io.Image(), output=bentoml.io.NumpyNdarray())
async def predict(f: PILImage) -> NDArray[t.Any]:
    arr = jnp.array(f) / 255.0
    arr = jnp.expand_dims(arr, (0, 3))
    res = await mnist_runner.async_run(arr)
    return res.argmax()
