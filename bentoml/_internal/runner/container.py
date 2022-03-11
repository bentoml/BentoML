import abc
import pickle
import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ..types import LazyType
from ..configuration.containers import DeploymentContainer

SingleType = t.TypeVar("SingleType")
BatchType = t.TypeVar("BatchType")

IndexType = t.Union[None, int]

if TYPE_CHECKING:
    from .. import external_typing as ext


class Payload(t.NamedTuple):
    data: bytes
    meta: t.Dict[str, t.Union[bool, int, float, str]]


class DataContainer(t.Generic[SingleType, BatchType]):
    @classmethod
    def create_payload(
        cls,
        data: bytes,
        meta: t.Optional[t.Dict[str, t.Union[bool, int, float, str]]] = None,
    ) -> Payload:
        return Payload(data, dict(meta or dict(), container=cls.__name__))

    @classmethod
    @abc.abstractmethod
    def singles_to_batch(
        cls, singles: t.Sequence[SingleType], batch_axis: int = 0
    ) -> BatchType:
        ...

    @classmethod
    @abc.abstractmethod
    def batch_to_singles(
        cls, batch: BatchType, batch_axis: int = 0
    ) -> t.List[SingleType]:
        ...

    @classmethod
    @abc.abstractmethod
    def single_to_payload(cls, single: SingleType) -> Payload:
        ...

    @classmethod
    @abc.abstractmethod
    def payload_to_single(cls, payload: Payload) -> SingleType:
        ...

    @classmethod
    @abc.abstractmethod
    def payload_to_batch(cls, payload: Payload) -> BatchType:
        ...

    @classmethod
    @abc.abstractmethod
    def batch_to_payload(cls, batch: BatchType) -> Payload:
        ...

    @classmethod
    def payloads_to_batch(
        cls, payload_list: t.Sequence[Payload], batch_axis: int = 0
    ) -> BatchType:
        return cls.singles_to_batch(
            [cls.payload_to_single(i) for i in payload_list], batch_axis=batch_axis
        )

    @classmethod
    def batch_to_payloads(
        cls, batch: BatchType, batch_axis: int = 0
    ) -> t.List[Payload]:
        return [
            cls.single_to_payload(i)
            for i in cls.batch_to_singles(batch, batch_axis=batch_axis)
        ]


class NdarrayContainer(
    DataContainer[
        "ext.NpNDArray",
        "ext.NpNDArray",
    ]
):
    @classmethod
    def singles_to_batch(
        cls,
        singles: t.Sequence["ext.NpNDArray"],
        batch_axis: int = 0,
    ) -> "ext.NpNDArray":
        import numpy as np

        return np.stack(  # type: ignore[reportGeneralTypeIssues]
            singles,
            axis=batch_axis,
        )

    @classmethod
    def batch_to_singles(
        cls,
        batch: "ext.NpNDArray",
        batch_axis: int = 0,
    ) -> t.List["ext.NpNDArray"]:
        import numpy as np

        return [
            np.squeeze(arr, axis=batch_axis)  # type: ignore
            for arr in np.split(  # type: ignore[list-item]
                batch,
                batch.shape[batch_axis],
                axis=batch_axis,
            )
        ]

    @classmethod
    @inject
    def single_to_payload(  # pylint: disable=arguments-differ
        cls,
        single: "ext.NpNDArray",
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
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
    def payload_to_single(  # pylint: disable=arguments-differ
        cls,
        payload: Payload,
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
    ) -> "ext.NpNDArray":
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            return plasma_db.get(plasma.ObjectID(payload.data))

        return pickle.loads(payload.data)

    batch_to_payload = single_to_payload  # type: ignore[assignment]
    payload_to_batch = payload_to_single


class PandasDataFrameContainer(
    DataContainer[t.Union["ext.PdDataFrame", "ext.PdSeries"], "ext.PdDataFrame"]
):
    @classmethod
    def singles_to_batch(
        cls,
        singles: t.Sequence[t.Union["ext.PdDataFrame", "ext.PdSeries"]],
        batch_axis: int = 0,
    ) -> "ext.PdDataFrame":
        import pandas as pd  # type: ignore[import]

        assert batch_axis == 0, "PandasDataFrameContainer requires batch_axis = 0"

        # here we assume each member of singles has the same type/shape
        head = singles[0]
        if LazyType["ext.PdDataFrame"](pd.DataFrame).isinstance(head):
            # DataFrame single type should only have one row
            assert (
                len(head) == 1
            ), "SingleType of PandasDataFrameContainer should have only one row"
            return pd.concat(singles)  # type: ignore[call-arg]

        # pd.Series
        return pd.concat(singles, axis=1).T  # type: ignore[arg-type]

    @classmethod
    def batch_to_singles(  # type: ignore[override]
        cls,
        batch: "ext.PdDataFrame",
        batch_axis: int = 0,
    ) -> t.List["ext.PdSeries"]:

        assert batch_axis == 0, "PandasDataFrameContainer requires batch_axis = 0"

        sers = [row for _, row in batch.iterrows()]
        return sers

    @classmethod
    @inject
    def single_to_payload(  # pylint: disable=arguments-differ
        cls,
        single: "t.Union[ext.PdDataFrame, ext.PdSeries]",
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
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
    def payload_to_single(  # pylint: disable=arguments-differ
        cls,
        payload: Payload,
        plasma_db: "ext.PlasmaClient" = Provide[DeploymentContainer.plasma_db],
    ):
        if payload.meta.get("plasma"):
            import pyarrow.plasma as plasma

            assert plasma_db
            return plasma_db.get(plasma.ObjectID(payload.data))

        return pickle.loads(payload.data)

    batch_to_payload = single_to_payload  # type: ignore[assignment]
    payload_to_batch = payload_to_single


class DefaultContainer(DataContainer[t.Any, t.List[t.Any]]):
    @classmethod
    def singles_to_batch(
        cls, singles: t.Sequence[t.Any], batch_axis: int = 0
    ) -> t.List[t.Any]:
        assert (
            batch_axis == 0
        ), "Default Runner DataContainer does not support batch_axies other than 0"
        return list(singles)

    @classmethod
    def batch_to_singles(
        cls, batch: t.List[t.Any], batch_axis: int = 0
    ) -> t.List[t.Any]:
        assert (
            batch_axis == 0
        ), "Default Runner DataContainer does not support batch_axies other than 0"
        return batch

    @classmethod
    def single_to_payload(cls, single: t.Any) -> Payload:
        if isinstance(single, t.Generator):  # Generators can't be pickled
            single = list(single)  # type: ignore
        return cls.create_payload(pickle.dumps(single))

    @classmethod
    @inject
    def payload_to_single(cls, payload: Payload):
        return pickle.loads(payload.data)

    batch_to_payload = single_to_payload  # type: ignore[assignment]
    payload_to_batch = payload_to_single


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
    def singles_to_batch(cls, singles: t.Any, batch_axis: int = 0):
        container_cls = DataContainerRegistry.find_by_single_type(type(singles[0]))
        return container_cls.singles_to_batch(singles, batch_axis=batch_axis)

    @classmethod
    def batch_to_singles(cls, batch: t.Any, batch_axis: int = 0) -> t.List[t.Any]:
        container_cls = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.batch_to_singles(batch, batch_axis=batch_axis)

    @classmethod
    def single_to_payload(cls, single: SingleType) -> Payload:  # type: ignore[override]
        container_cls = DataContainerRegistry.find_by_single_type(type(single))
        return container_cls.single_to_payload(single)

    @classmethod
    def payload_to_single(
        cls,
        payload: Payload,
    ) -> SingleType:  # type: ignore[override]
        container_cls = DataContainerRegistry.find_by_name(
            str(payload.meta.get("container"))
        )
        return container_cls.payload_to_single(payload)

    @classmethod
    def payload_to_batch(cls, payload: Payload) -> BatchType:  # type: ignore[override]
        container_cls = DataContainerRegistry.find_by_name(
            str(payload.meta.get("container"))
        )
        return container_cls.payload_to_batch(payload)

    @classmethod
    @abc.abstractmethod
    def batch_to_payload(cls, batch: BatchType) -> Payload:  # type: ignore[override]
        container_cls = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.batch_to_payload(batch)

    @classmethod
    def payloads_to_batch(cls, payload_list: t.Sequence[Payload], batch_axis: int = 0):
        container_cls = DataContainerRegistry.find_by_name(
            str(payload_list[0].meta.get("container"))
        )
        return container_cls.payloads_to_batch(payload_list, batch_axis=batch_axis)

    @classmethod
    def batch_to_payloads(
        cls,
        batch: BatchType,  # type: ignore[override]
        batch_axis: int = 0,
    ) -> t.List[Payload]:
        container_cls = DataContainerRegistry.find_by_batch_type(type(batch))
        return container_cls.batch_to_payloads(batch, batch_axis=batch_axis)
