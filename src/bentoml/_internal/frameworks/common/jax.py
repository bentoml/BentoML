from __future__ import annotations

import pickle
import typing as t
import itertools
from typing import TYPE_CHECKING

from simple_di import inject

from ...types import LazyType
from ...utils import LazyLoader
from ....exceptions import MissingDependencyException
from ...runner.container import Payload
from ...runner.container import DataContainer
from ...runner.container import DataContainerRegistry

try:
    import jaxlib as jaxlib
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "jax requires jaxlib to be installed. See https://github.com/google/jax#installation for installation instructions."
    ) from None

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "jax is required in order to use with 'bentoml.flax'. See https://github.com/google/jax#installation for installation instructions."
    ) from None

if TYPE_CHECKING:
    import numpy as np

else:
    np = LazyLoader("numpy", globals(), "numpy")

__all__ = ["jax", "jnp", "jaxlib", "JaxArrayContainer"]


class JaxArrayContainer(DataContainer[jax.Array, jax.Array]):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence[jax.Array],
        batch_dim: int = 0,
    ) -> tuple[jax.Array, list[int]]:
        batch: jax.Array = jnp.concatenate(batches, axis=batch_dim)
        indices: list[int] = list(
            itertools.accumulate(subbatch.shape[0] for subbatch in batches)
        )
        indices = [0] + indices
        return batch, indices

    @classmethod
    def batch_to_batches(
        cls,
        batch: jax.Array,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> list[jax.Array]:
        return jnp.split(batch, indices[1:-1], axis=batch_dim)

    @classmethod
    @inject
    def to_payload(
        cls,
        batch: jax.Array,
        batch_dim: int = 0,
    ) -> Payload:
        return cls.create_payload(
            pickle.dumps(np.asarray(batch)),
            batch.shape[batch_dim],
        )

    @classmethod
    @inject
    def from_payload(
        cls,
        payload: Payload,
    ) -> jax.Array:
        return jnp.asarray(pickle.loads(payload.data))

    @classmethod
    @inject
    def batch_to_payloads(
        cls,
        batch: jax.Array,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> t.List[Payload]:
        batches = cls.batch_to_batches(batch, indices, batch_dim)
        payloads = [cls.to_payload(subbatch, batch_dim) for subbatch in batches]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
    ) -> tuple[jax.Array, list[int]]:
        batches = [cls.from_payload(payload) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


DataContainerRegistry.register_container(
    LazyType("jax.numpy", "ndarray"),
    LazyType("jax.numpy", "ndarray"),
    JaxArrayContainer,
)

DataContainerRegistry.register_container(
    LazyType("jax", "Array"),
    LazyType("jax", "Array"),
    JaxArrayContainer,
)
