from __future__ import annotations

import abc
import pickle
import typing as t
import itertools
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ..types import LazyType
from ..configuration.containers import DeploymentContainer

SingleType = t.TypeVar("SingleType")
BatchType = t.TypeVar("BatchType")
BatchIndicesType = list[int]

if TYPE_CHECKING:
    from .. import external_typing as ext


class Payload(t.NamedTuple):
    data: bytes
    meta: dict[str, bool | int | float | str]
    container: str


class DataContainer(t.Generic[SingleType, BatchType]):
    @classmethod
    def create_payload(
        cls,
        data: bytes,
        meta: t.Optional[t.Dict[str, t.Union[bool, int, float, str]]] = None,
    ) -> Payload:
        return Payload(data, meta or {}, container=cls.__name__)

    @classmethod
    @abc.abstractmethod
    def to_payload(cls, single: SingleType) -> Payload:
        ...

    @classmethod
    @abc.abstractmethod
    def from_payload(cls, payload: Payload) -> SingleType:
        ...

    @classmethod
    @abc.abstractmethod
    def batches_to_batch(
        cls, batches: t.Sequence[BatchType], batch_dim: int
    ) -> t.Tuple[BatchType, BatchIndicesType]:
        ...

    @classmethod
    @abc.abstractmethod
    def batch_to_batches(
        cls, batch: BatchType, indices: t.Sequence[int], batch_dim: int
    ) -> t.List[BatchType]:
        ...

    @classmethod
    @abc.abstractmethod
    def batch_to_payloads(
        cls, batch: BatchType, indices: t.Sequence[int], batch_dim: int
    ) -> t.List[Payload]:
        ...

    @classmethod
    @abc.abstractmethod
    def from_batch_payloads(
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int,
    ) -> tuple[BatchType, BatchIndicesType]:
        ...


class NdarrayContainer(
    DataContainer[
        "ext.NpNDArray",
        "ext.NpNDArray",
    ]
):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence["ext.NpNDArray"],
        batch_dim: int = 0,
    ) -> t.Tuple["ext.NpNDArray", BatchIndicesType]:
        import numpy as np

        # numpy.concatenate may consume lots of memory, need optimization later
        batch = np.concatenate(  # type: ignore[reportGeneralTypeIssues]
            batches,
            axis=batch_dim,
        )
        indices = list(
            itertools.accumulate(subbatch.shape[batch_dim] for subbatch in batches)
        )
        indices = [0] + indices
        return batch, indices

    @classmethod
    def batch_to_batches(
        cls,
        batch: "ext.NpNDArray",
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> t.List["ext.NpNDArray"]:
        import numpy as np

        return np.split(batch, indices[1:-1], axis=batch_dim)

    @classmethod
    @inject
    def to_payload(  # pylint: disable=arguments-differ
        cls,
        single: "ext.NpNDArray",
        plasma_db: "ext.PlasmaClient" | None = Provide[DeploymentContainer.plasma_db],
    ) -> Payload:
        if plasma_db:
            return cls.create_payload(
                plasma_db.put(single).binary(),
                {"plasma": True},
            )

        return cls.create_payload(
            pickle.dumps(single),
            {"plasma": False},
        )

    @classmethod
    @inject
    def from_payload(  # pylint: disable=arguments-differ
        cls,
        payload: Payload,
        plasma_db: "ext.PlasmaClient" | None = Provide[DeploymentContainer.plasma_db],
    ) -> "ext.NpNDArray":
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            return plasma_db.get(plasma.ObjectID(payload.data))

        return pickle.loads(payload.data)

    @classmethod
    @inject
    def batch_to_payloads(  # pylint: disable=arguments-differ
        cls,
        batch: "ext.NpNDArray",
        indices: t.Sequence[int],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
    ) -> t.List[Payload]:

        batches = cls.batch_to_batches(batch, indices, batch_dim)

        def to_payload(subbatch: "ext.NpNDArray"):
            payload = cls.to_payload(subbatch, plasma_db)
            payload.meta["batch_size"] = subbatch.shape[batch_dim]
            return payload

        payloads = [to_payload(subbatch) for subbatch in batches]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(  # pylint: disable=arguments-differ
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
    ) -> t.Tuple["ext.NpNDArray", BatchIndicesType]:
        batches = [cls.from_payload(payload, plasma_db) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


class PandasDataFrameContainer(
    DataContainer[t.Union["ext.PdDataFrame", "ext.PdSeries"], "ext.PdDataFrame"]
):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence["ext.PdDataFrame"],
        batch_dim: int = 0,
    ) -> t.Tuple["ext.PdDataFrame", BatchIndicesType]:
        import pandas as pd  # type: ignore[import]

        assert (
            batch_dim == 0
        ), "PandasDataFrameContainer does not support batch_dim other than 0"
        indices = list(
            itertools.accumulate(subbatch.shape[batch_dim] for subbatch in batches)
        )
        indices = [0] + indices
        return pd.concat(batches, ignore_index=True), indices

    @classmethod
    def batch_to_batches(  # type: ignore[override]
        cls,
        batch: "ext.PdDataFrame",
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> t.List["ext.PdDataFrame"]:

        assert (
            batch_dim == 0
        ), "PandasDataFrameContainer does not support batch_dim other than 0"

        return [
            batch.iloc[indices[i] : indices[i + 1]].reset_index(drop=True)
            for i in range(len(indices) - 1)
        ]

    @classmethod
    @inject
    def to_payload(  # pylint: disable=arguments-differ
        cls,
        single: "t.Union[ext.PdDataFrame, ext.PdSeries]",
        plasma_db: "ext.PlasmaClient" | None = Provide[DeploymentContainer.plasma_db],
    ) -> Payload:
        if plasma_db:
            return cls.create_payload(
                plasma_db.put(single).binary(),
                {"plasma": True},
            )

        return cls.create_payload(
            pickle.dumps(single),
            {"plasma": False},
        )

    @classmethod
    @inject
    def from_payload(  # pylint: disable=arguments-differ
        cls,
        payload: Payload,
        plasma_db: "ext.PlasmaClient" | None = Provide[DeploymentContainer.plasma_db],
    ):
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            return plasma_db.get(plasma.ObjectID(payload.data))

        return pickle.loads(payload.data)

    @classmethod
    @inject
    def batch_to_payloads(  # pylint: disable=arguments-differ
        cls,
        batch: "ext.PdDataFrame",
        indices: t.Sequence[int],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
    ) -> t.List[Payload]:

        batches = cls.batch_to_batches(batch, indices, batch_dim)

        def to_payload(subbatch: "ext.PdDataFrame"):
            payload = cls.to_payload(subbatch, plasma_db)
            payload.meta["batch_size"] = subbatch.shape[batch_dim]
            return payload

        payloads = [to_payload(subbatch) for subbatch in batches]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(  # pylint: disable=arguments-differ
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
    ) -> t.Tuple["ext.PdDataFrame", BatchIndicesType]:
        batches = [cls.from_payload(payload, plasma_db) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


class DefaultContainer(DataContainer[t.Any, t.List[t.Any]]):
    @classmethod
    def batches_to_batch(
        cls, batches: t.Sequence[t.List[t.Any]], batch_dim: int = 0
    ) -> t.Tuple[t.List[t.Any], BatchIndicesType]:
        assert (
            batch_dim == 0
        ), "Default Runner DataContainer does not support batch_dim other than 0"
        batch = []
        for subbatch in batches:
            batch.extend(subbatch)
        indices = list(itertools.accumulate(len(subbatch) for subbatch in batches))
        indices = [0] + indices
        return batch, indices  # type: ignore[reportUnknowVariableType]

    @classmethod
    def batch_to_batches(
        cls, batch: t.List[t.Any], indices: t.Sequence[int], batch_dim: int = 0
    ) -> t.List[t.List[t.Any]]:
        assert (
            batch_dim == 0
        ), "Default Runner DataContainer does not support batch_dim other than 0"
        return [batch[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]

    @classmethod
    def to_payload(cls, single: t.Any) -> Payload:
        if isinstance(single, t.Generator):  # Generators can't be pickled
            single = list(single)  # type: ignore
        return cls.create_payload(pickle.dumps(single))

    @classmethod
    @inject
    def from_payload(cls, payload: Payload):
        return pickle.loads(payload.data)

    @classmethod
    @inject
    def batch_to_payloads(
        cls,
        batch: t.List[t.Any],
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> t.List[Payload]:

        batches = cls.batch_to_batches(batch, indices, batch_dim)

        def to_payload(subbatch: t.List[t.Any]):
            payload = cls.to_payload(subbatch)
            payload.meta["batch_size"] = len(subbatch)
            return payload

        payloads = [to_payload(subbatch) for subbatch in batches]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
    ) -> t.Tuple[t.List[t.Any], BatchIndicesType]:
        batches = [cls.from_payload(payload) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


class DataContainerRegistry:
    CONTAINER_SINGLE_TYPE_MAP: t.Dict[
        LazyType[t.Any], t.Type[DataContainer[t.Any, t.Any]]
    ] = dict()
    CONTAINER_BATCH_TYPE_MAP: t.Dict[
        LazyType[t.Any], t.Type[DataContainer[t.Any, t.Any]]
    ] = dict()

    @classmethod
    def register_container(
        cls,
        single_type: t.Union[LazyType[t.Any], type],
        batch_type: t.Union[LazyType[t.Any], type],
        container_cls: t.Type[DataContainer[t.Any, t.Any]],
    ):
        single_type = LazyType.from_type(single_type)
        batch_type = LazyType.from_type(batch_type)

        cls.CONTAINER_BATCH_TYPE_MAP[batch_type] = container_cls
        cls.CONTAINER_SINGLE_TYPE_MAP[single_type] = container_cls

    @classmethod
    def find_by_single_type(
        cls, type_: t.Union[t.Type[SingleType], LazyType[t.Any]]
    ) -> t.Type[DataContainer[SingleType, BatchType]]:  # type: ignore[override]
        typeref = LazyType.from_type(type_)
        return cls.CONTAINER_SINGLE_TYPE_MAP.get(
            typeref,
            DefaultContainer,
        )  # type: ignore[arg-type]

    @classmethod
    def find_by_batch_type(
        cls, type_: t.Union[t.Type[BatchType], LazyType[t.Any]]
    ) -> t.Type[DataContainer[SingleType, BatchType]]:  # type: ignore[override]
        typeref = LazyType.from_type(type_)
        return cls.CONTAINER_BATCH_TYPE_MAP.get(
            typeref,
            DefaultContainer,
        )  # type: ignore[arg-type]

    @classmethod
    def find_by_name(cls, name: str) -> t.Type[DataContainer[t.Any, t.Any]]:
        for container_cls in cls.CONTAINER_BATCH_TYPE_MAP.values():
            if container_cls.__name__ == name:
                return container_cls
        if name == DefaultContainer.__name__:
            return DefaultContainer
        raise ValueError(f"can not find specified container class by name {name}")


def register_builtin_containers():
    DataContainerRegistry.register_container(
        LazyType("numpy", "ndarray"), LazyType("numpy", "ndarray"), NdarrayContainer
    )
    # DataContainerRegistry.register_container(np.ndarray, np.ndarray, NdarrayContainer)

    DataContainerRegistry.register_container(
        LazyType("pandas.core.series", "Series"),
        LazyType("pandas.core.frame", "DataFrame"),
        PandasDataFrameContainer,
    )

    DataContainerRegistry.register_container(
        LazyType("pandas.core.frame", "DataFrame"),
        LazyType("pandas.core.frame", "DataFrame"),
        PandasDataFrameContainer,
    )


register_builtin_containers()


class AutoContainer(DataContainer[t.Any, t.Any]):
    @classmethod
    def to_payload(cls, single: t.Any) -> Payload:
        container_cls = DataContainerRegistry.find_by_single_type(type(single))
        return container_cls.to_payload(single)

    @classmethod
    def from_payload(cls, payload: Payload) -> tuple[t.Any, list[int]]:
        container_cls = DataContainerRegistry.find_by_name(payload.container)
        return container_cls.from_payload(payload)

    @classmethod
    def batches_to_batch(
        cls, batches: t.Sequence[BatchType], batch_dim: int = 0
    ) -> tuple[BatchType, BatchIndicesType]:
        container_cls = DataContainerRegistry.find_by_batch_type(type(batches[0]))
        return container_cls.batches_to_batch(batches, batch_dim)

    @classmethod
    def batch_to_batches(
        cls, batch: BatchType, indices: t.Sequence[int], batch_dim: int = 0
    ) -> list[BatchType]:
        container_cls = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.batch_to_batches(batch, indices, batch_dim)

    @classmethod
    def batch_to_payloads(
        cls,
        batch: t.Any,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> t.List[Payload]:
        container_cls = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.batch_to_payloads(batch, indices, batch_dim)

    @classmethod
    def from_batch_payloads(
        cls, payloads: t.Sequence[Payload], batch_dim: int = 0
    ) -> tuple[t.Any, BatchIndicesType]:
        container_cls = DataContainerRegistry.find_by_name(payloads[0].container)
        return container_cls.from_batch_payloads(payloads, batch_dim)
