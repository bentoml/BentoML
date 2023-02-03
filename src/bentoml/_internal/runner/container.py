from __future__ import annotations

import abc
import pickle
import typing as t
import itertools

from simple_di import inject
from simple_di import Provide

from ..types import LazyType
from ..utils import LazyLoader
from ..configuration.containers import BentoMLContainer

SingleType = t.TypeVar("SingleType")
BatchType = t.TypeVar("BatchType")

if t.TYPE_CHECKING:
    import tritonclient.grpc.aio as tritongrpcclient

    from .. import external_typing as ext
else:
    tritongrpcclient = LazyLoader(
        "tritongrpcclient",
        globals(),
        "tritonclient.grpc.aio",
        exc_msg="tritonclient is required to use triton with BentoML. Install with 'pip install \"tritonclient[grpc]>=2.29.0\"'.",
    )


class Payload(t.NamedTuple):
    data: bytes
    meta: dict[str, bool | int | float | str]
    container: str
    batch_size: int = -1


class DataContainer(t.Generic[SingleType, BatchType]):
    @classmethod
    def create_payload(
        cls,
        data: bytes,
        batch_size: int,
        meta: dict[str, bool | int | float | str] | None = None,
    ) -> Payload:
        return Payload(data, meta or {}, container=cls.__name__, batch_size=batch_size)

    @classmethod
    @abc.abstractmethod
    def to_payload(cls, batch: BatchType, batch_dim: int) -> Payload:
        ...

    @classmethod
    @abc.abstractmethod
    def from_payload(cls, payload: Payload) -> BatchType:
        ...

    @classmethod
    def to_triton_payload(
        cls,
        inp: SingleType,
        meta: tritongrpcclient.service_pb2.ModelMetadataResponse.TensorMetadata,
    ) -> tritongrpcclient.InferInput:
        """
        Convert given input types to a Triton payload.

        Implementation of this method should always return a Payload that has
        data of a Numpy array.
        """
        raise NotImplementedError(
            f"{cls.__name__} doesn't support converting to Triton payload."
        )

    @classmethod
    @abc.abstractmethod
    def batches_to_batch(
        cls, batches: t.Sequence[BatchType], batch_dim: int
    ) -> tuple[BatchType, list[int]]:
        ...

    @classmethod
    @abc.abstractmethod
    def batch_to_batches(
        cls, batch: BatchType, indices: t.Sequence[int], batch_dim: int
    ) -> list[BatchType]:
        ...

    @classmethod
    @abc.abstractmethod
    def batch_to_payloads(
        cls, batch: BatchType, indices: t.Sequence[int], batch_dim: int
    ) -> list[Payload]:
        ...

    @classmethod
    @abc.abstractmethod
    def from_batch_payloads(
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int,
    ) -> tuple[BatchType, list[int]]:
        ...


class TritonInferInputDataContainer(
    DataContainer["tritongrpcclient.InferInput", "tritongrpcclient.InferInput"]
):
    @classmethod
    def to_payload(cls, batch: tritongrpcclient.InferInput, batch_dim: int) -> Payload:
        raise NotImplementedError

    @classmethod
    def from_payload(cls, payload: Payload) -> tritongrpcclient.InferInput:
        raise NotImplementedError

    @classmethod
    def to_triton_payload(
        cls,
        inp: tritongrpcclient.InferInput,
        meta: tritongrpcclient.service_pb2.ModelMetadataResponse.TensorMetadata,
    ) -> tritongrpcclient.InferInput:
        return inp

    @classmethod
    @abc.abstractmethod
    def batches_to_batch(
        cls, batches: t.Sequence[tritongrpcclient.InferInput], batch_dim: int
    ) -> tuple[tritongrpcclient.InferInput, list[int]]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def batch_to_batches(
        cls,
        batch: tritongrpcclient.InferInput,
        indices: t.Sequence[int],
        batch_dim: int,
    ) -> list[tritongrpcclient.InferInput]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def batch_to_payloads(
        cls,
        batch: tritongrpcclient.InferInput,
        indices: t.Sequence[int],
        batch_dim: int,
    ) -> list[Payload]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_batch_payloads(
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int,
    ) -> tuple[tritongrpcclient.InferInput, list[int]]:
        raise NotImplementedError


class NdarrayContainer(DataContainer["ext.NpNDArray", "ext.NpNDArray"]):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence[ext.NpNDArray],
        batch_dim: int = 0,
    ) -> tuple[ext.NpNDArray, list[int]]:
        import numpy as np

        # numpy.concatenate may consume lots of memory, need optimization later
        batch: ext.NpNDArray = np.concatenate(  # type: ignore  (no numpy types)
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
        batch: ext.NpNDArray,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> list[ext.NpNDArray]:
        import numpy as np

        return np.split(batch, indices[1:-1], axis=batch_dim)  # type: ignore  (no numpy types)

    @classmethod
    def to_triton_payload(
        cls,
        inp: ext.NpNDArray,
        meta: tritongrpcclient.service_pb2.ModelMetadataResponse.TensorMetadata,
    ) -> tritongrpcclient.InferInput:
        InferInput = tritongrpcclient.InferInput(
            meta.name, inp.shape, tritongrpcclient.np_to_triton_dtype(inp.dtype)
        )
        InferInput.set_data_from_numpy(inp)
        return InferInput

    @classmethod
    @inject
    def to_payload(
        cls,
        batch: ext.NpNDArray,
        batch_dim: int,
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
    def from_payload(
        cls,
        payload: Payload,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> ext.NpNDArray:
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            return plasma_db.get(plasma.ObjectID(payload.data))

        return pickle.loads(payload.data)

    @classmethod
    @inject
    def batch_to_payloads(
        cls,
        batch: ext.NpNDArray,
        indices: t.Sequence[int],
        batch_dim: int = 0,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> list[Payload]:

        batches = cls.batch_to_batches(batch, indices, batch_dim)

        payloads = [
            cls.to_payload(subbatch, batch_dim, plasma_db) for subbatch in batches
        ]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
        plasma_db: "ext.PlasmaClient" | None = Provide[BentoMLContainer.plasma_db],
    ) -> t.Tuple["ext.NpNDArray", list[int]]:
        batches = [cls.from_payload(payload, plasma_db) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


class PandasDataFrameContainer(
    DataContainer[t.Union["ext.PdDataFrame", "ext.PdSeries"], "ext.PdDataFrame"]
):
    @classmethod
    def batches_to_batch(
        cls,
        batches: t.Sequence[ext.PdDataFrame],
        batch_dim: int = 0,
    ) -> tuple[ext.PdDataFrame, list[int]]:
        import pandas as pd

        assert (
            batch_dim == 0
        ), "PandasDataFrameContainer does not support batch_dim other than 0"
        indices = list(
            itertools.accumulate(subbatch.shape[batch_dim] for subbatch in batches)
        )
        indices = [0] + indices
        return pd.concat(batches, ignore_index=True), indices  # type: ignore (incomplete panadas types)

    @classmethod
    def batch_to_batches(
        cls,
        batch: ext.PdDataFrame,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> list[ext.PdDataFrame]:

        assert (
            batch_dim == 0
        ), "PandasDataFrameContainer does not support batch_dim other than 0"

        return [
            batch.iloc[indices[i] : indices[i + 1]].reset_index(drop=True)
            for i in range(len(indices) - 1)
        ]

    @classmethod
    @inject
    def to_payload(
        cls,
        batch: ext.PdDataFrame | ext.PdSeries,
        batch_dim: int,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> Payload:
        import pandas as pd

        assert (
            batch_dim == 0
        ), "PandasDataFrameContainer does not support batch_dim other than 0"

        if isinstance(batch, pd.Series):
            batch = pd.DataFrame([batch])

        if plasma_db:
            return cls.create_payload(
                plasma_db.put(batch).binary(),
                batch.size,
                {"plasma": True},
            )

        return cls.create_payload(
            pickle.dumps(batch),
            batch.size,
            {"plasma": False},
        )

    @classmethod
    @inject
    def from_payload(
        cls,
        payload: Payload,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> ext.PdDataFrame:
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            return plasma_db.get(plasma.ObjectID(payload.data))

        return pickle.loads(payload.data)

    @classmethod
    @inject
    def batch_to_payloads(
        cls,
        batch: ext.PdDataFrame,
        indices: t.Sequence[int],
        batch_dim: int = 0,
        plasma_db: ext.PlasmaClient | None = Provide[BentoMLContainer.plasma_db],
    ) -> list[Payload]:

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
    ) -> tuple[ext.PdDataFrame, list[int]]:
        batches = [cls.from_payload(payload, plasma_db) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


class DefaultContainer(DataContainer[t.Any, t.List[t.Any]]):
    @classmethod
    def batches_to_batch(
        cls, batches: t.Sequence[list[t.Any]], batch_dim: int = 0
    ) -> tuple[list[t.Any], list[int]]:
        assert (
            batch_dim == 0
        ), "Default Runner DataContainer does not support batch_dim other than 0"
        batch: list[t.Any] = []
        for subbatch in batches:
            batch.extend(subbatch)
        indices = list(itertools.accumulate(len(subbatch) for subbatch in batches))
        indices = [0] + indices
        return batch, indices

    @classmethod
    def batch_to_batches(
        cls, batch: list[t.Any], indices: t.Sequence[int], batch_dim: int = 0
    ) -> list[list[t.Any]]:
        assert (
            batch_dim == 0
        ), "Default Runner DataContainer does not support batch_dim other than 0"
        return [batch[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]

    @classmethod
    def to_payload(cls, batch: t.Any, batch_dim: int) -> Payload:
        if isinstance(batch, t.Generator):  # Generators can't be pickled
            batch = list(t.cast(t.Generator[t.Any, t.Any, t.Any], batch))
        if isinstance(batch, list):
            return cls.create_payload(
                pickle.dumps(batch), len(t.cast(t.List[t.Any], batch))
            )
        else:
            return cls.create_payload(pickle.dumps(batch), 1)

    @classmethod
    @inject
    def from_payload(cls, payload: Payload) -> t.Any:
        return pickle.loads(payload.data)

    @classmethod
    @inject
    def batch_to_payloads(
        cls,
        batch: list[t.Any],
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> list[Payload]:

        batches = cls.batch_to_batches(batch, indices, batch_dim)

        payloads = [cls.to_payload(subbatch, batch_dim) for subbatch in batches]
        return payloads

    @classmethod
    @inject
    def from_batch_payloads(
        cls,
        payloads: t.Sequence[Payload],
        batch_dim: int = 0,
    ) -> tuple[list[t.Any], list[int]]:
        batches = [cls.from_payload(payload) for payload in payloads]
        return cls.batches_to_batch(batches, batch_dim)


class DataContainerRegistry:
    CONTAINER_SINGLE_TYPE_MAP: t.Dict[
        LazyType[t.Any], t.Type[DataContainer[t.Any, t.Any]]
    ] = dict()
    CONTAINER_BATCH_TYPE_MAP: t.Dict[
        LazyType[t.Any], type[DataContainer[t.Any, t.Any]]
    ] = dict()

    @classmethod
    def register_container(
        cls,
        single_type: LazyType[t.Any] | type,
        batch_type: LazyType[t.Any] | type,
        container_cls: t.Type[DataContainer[t.Any, t.Any]],
    ):
        single_type = LazyType.from_type(single_type)
        batch_type = LazyType.from_type(batch_type)

        cls.CONTAINER_BATCH_TYPE_MAP[batch_type] = container_cls
        cls.CONTAINER_SINGLE_TYPE_MAP[single_type] = container_cls

    @classmethod
    def find_by_single_type(
        cls, type_: t.Type[SingleType] | LazyType[SingleType]
    ) -> t.Type[DataContainer[SingleType, t.Any]]:
        typeref = LazyType.from_type(type_)
        return cls.CONTAINER_SINGLE_TYPE_MAP.get(
            typeref,
            DefaultContainer,
        )

    @classmethod
    def find_by_batch_type(
        cls, type_: t.Type[BatchType] | LazyType[BatchType]
    ) -> t.Type[DataContainer[t.Any, BatchType]]:
        typeref = LazyType.from_type(type_)
        return cls.CONTAINER_BATCH_TYPE_MAP.get(
            typeref,
            DefaultContainer,
        )

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

    DataContainerRegistry.register_container(
        LazyType("tritonclient.grpc", "InferInput"),
        LazyType("tritonclient.grpc", "InferInput"),
        TritonInferInputDataContainer,
    )
    DataContainerRegistry.register_container(
        LazyType("tritonclient.grpc.aio", "InferInput"),
        LazyType("tritonclient.grpc.aio", "InferInput"),
        TritonInferInputDataContainer,
    )


register_builtin_containers()


class AutoContainer(DataContainer[t.Any, t.Any]):
    @classmethod
    def to_payload(cls, batch: t.Any, batch_dim: int) -> Payload:
        container_cls: t.Type[
            DataContainer[t.Any, t.Any]
        ] = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.to_payload(batch, batch_dim)

    @classmethod
    def from_payload(cls, payload: Payload) -> t.Any:
        container_cls = DataContainerRegistry.find_by_name(payload.container)
        return container_cls.from_payload(payload)

    @classmethod
    def to_triton_payload(
        cls,
        inp: t.Any,
        meta: tritongrpcclient.service_pb2.ModelMetadataResponse.TensorMetadata,
    ) -> tritongrpcclient.InferInput:
        return DataContainerRegistry.find_by_single_type(type(inp)).to_triton_payload(
            inp, meta
        )

    @classmethod
    def batches_to_batch(
        cls, batches: t.Sequence[BatchType], batch_dim: int = 0
    ) -> tuple[BatchType, list[int]]:
        container_cls: t.Type[
            DataContainer[t.Any, t.Any]
        ] = DataContainerRegistry.find_by_batch_type(type(batches[0]))
        return container_cls.batches_to_batch(batches, batch_dim)

    @classmethod
    def batch_to_batches(
        cls, batch: BatchType, indices: t.Sequence[int], batch_dim: int = 0
    ) -> list[BatchType]:
        container_cls: t.Type[
            DataContainer[t.Any, t.Any]
        ] = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.batch_to_batches(batch, indices, batch_dim)

    @classmethod
    def batch_to_payloads(
        cls,
        batch: t.Any,
        indices: t.Sequence[int],
        batch_dim: int = 0,
    ) -> list[Payload]:
        container_cls: t.Type[
            DataContainer[t.Any, t.Any]
        ] = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.batch_to_payloads(batch, indices, batch_dim)

    @classmethod
    def from_batch_payloads(
        cls, payloads: t.Sequence[Payload], batch_dim: int = 0
    ) -> tuple[t.Any, list[int]]:
        container_cls = DataContainerRegistry.find_by_name(payloads[0].container)
        return container_cls.from_batch_payloads(payloads, batch_dim)
