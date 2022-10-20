from __future__ import annotations

import typing as t

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
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "jax is required in order to use with 'bentoml.flax'. See https://github.com/google/jax#installation for installation instructions."
    ) from None


class JaxArrayContainer(DataContainer[jnp.ndarray, jnp.ndarray]):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence[torch.Tensor],
        batch_dim: int = 0,
    ) -> tuple[jnp.ndarray, list[int]]:
        batch = torch.cat(tuple(batches), dim=batch_dim)
        indices = list(
            itertools.accumulate(subbatch.shape[batch_dim] for subbatch in batches)
        )
        indices = [0] + indices
        return batch, indices

    @classmethod
    def batch_to_batches(
        cls,
        batch: torch.Tensor,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> t.List[torch.Tensor]:
        sizes = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
        output: list[torch.Tensor] = torch.split(batch, sizes, dim=batch_dim)
        return output

    @classmethod
    @inject
    def to_payload(  # pylint: disable=arguments-differ
        cls,
        batch: torch.Tensor,
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> Payload:
        batch = batch.cpu().numpy()
        if plasma_db:
            return cls.create_payload(
                plasma_db.put(batch).binary(),
                batch_size=batch.shape[batch_dim],
                meta={"plasma": True},
            )

        return cls.create_payload(
            pickle.dumps(batch),
            batch_size=batch.shape[batch_dim],
            meta={"plasma": False},
        )

    @classmethod
    @inject
    def from_payload(  # pylint: disable=arguments-differ
        cls,
        payload: Payload,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> torch.Tensor:
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            ret = plasma_db.get(plasma.ObjectID(payload.data))

        else:
            ret = pickle.loads(payload.data)
        return torch.from_numpy(ret).requires_grad_(False)

    @classmethod
    @inject
    def batch_to_payloads(  # pylint: disable=arguments-differ
        cls,
        batch: torch.Tensor,
        indices: t.Sequence[int],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> t.List[Payload]:
        batches = cls.batch_to_batches(batch, indices, batch_dim)
        payloads = [cls.to_payload(i, batch_dim=batch_dim) for i in batches]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(  # pylint: disable=arguments-differ
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> t.Tuple[torch.Tensor, list[int]]:
        batches = [cls.from_payload(payload, plasma_db) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


DataContainerRegistry.register_container(
    LazyType("jax.numpy", "ndarray"),
    LazyType("jax.numpy", "ndarray"),
    JaxArrayContainer,
)
