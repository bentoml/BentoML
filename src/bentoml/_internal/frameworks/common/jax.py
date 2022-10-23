from __future__ import annotations

import pickle
import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ...types import LazyType
from ....exceptions import MissingDependencyException
from ...runner.container import Payload
from ...runner.container import DataContainer
from ...runner.container import DataContainerRegistry
from ...configuration.containers import BentoMLContainer

try:
    import jaxlib as jaxlib  # type: ignore (early check) # pylint: disable=unused-import
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "jax requires jaxlib to be installed. See https://github.com/google/jax#installation for installation instructions."
    ) from None

try:
    import jax  # type: ignore (early check) # pylint: disable=unused-import
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "jax is required in order to use with 'bentoml.flax'. See https://github.com/google/jax#installation for installation instructions."
    ) from None

if TYPE_CHECKING:
    from ... import external_typing as ext


class JaxArrayContainer(DataContainer[jnp.ndarray, jnp.ndarray]):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence[jnp.ndarray],
        batch_dim: int = 0,
    ) -> tuple[jnp.ndarray, list[int]]:
        batch: jnp.ndarray = jnp.concatenate(batches, axis=batch_dim)
        indices: list[int] = list(
            itertools.accumulate(subbatch.shape[0] for subbatch in batches)
        )
        indices = [0] + indices
        return batch, indices

    @classmethod
    def batch_to_batches(
        cls,
        batch: jnp.ndarray,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> list[jnp.ndarray]:
        return jnp.split(batch, indices[1:-1], axis=batch_dim)

    @classmethod
    @inject
    def to_payload(  # pylint: disable=arguments-differ
        cls,
        batch: jnp.ndarray,
        batch_dim: int = 0,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> Payload:
        if plasma_db:
            return cls.create_payload(
                plasma_db.put(batch).binary(),
                batch.shape[batch_dim],
                {"plasma": True},
            )

        return cls.create_payload(
            pickle.dumps(batch),
            batch.shape[batch_dim],
            {"plasma": False},
        )

    @classmethod
    @inject
    def from_payload(  # pylint: disable=arguments-differ
        cls,
        payload: Payload,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> jnp.ndarray:
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            return plasma_db.get(plasma.ObjectID(payload.data))
        return pickle.loads(payload.data)

    @classmethod
    @inject
    def batch_to_payloads(  # pylint: disable=arguments-differ
        cls,
        batch: jnp.ndarray,
        indices: t.Sequence[int],
        batch_dim: int = 0,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> t.List[Payload]:
        batches = cls.batch_to_batches(batch, indices, batch_dim)
        payloads = [
            cls.to_payload(subbatch, batch_dim, plasma_db) for subbatch in batches
        ]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(  # pylint: disable=arguments-differ
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> tuple[jnp.ndarray, list[int]]:
        batches = [cls.from_payload(payload, plasma_db) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


DataContainerRegistry.register_container(
    LazyType("jax.numpy", "ndarray"),
    LazyType("jax.numpy", "ndarray"),
    JaxArrayContainer,
)
