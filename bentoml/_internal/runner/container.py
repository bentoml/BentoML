import abc
import typing as t
from typing import TYPE_CHECKING

import cloudpickle
from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer

from .utils import TypeRef

SingleType = t.TypeVar("SingleType")
BatchType = t.TypeVar("BatchType")

IndexType = t.Union[None, int]

if TYPE_CHECKING:
    import numpy as np


class Payload(t.NamedTuple):
    datas: t.List[bytes]
    meta: t.Dict[str, t.Union[bool, int, float, str]]


class DataContainer(t.Generic[SingleType, BatchType]):
    def __init__(self, datas: t.List[SingleType] = None) -> None:
        self._datas = datas or []

    def append(self, data: SingleType) -> None:
        self._datas.append(data)

    def __getitem__(self, key) -> SingleType:
        return self._datas[key]

    @abc.abstractmethod
    def to_batch(self, batch_axis=0) -> BatchType:
        ...

    @classmethod
    def merge(cls, insts: t.List["DataContainer"]) -> "DataContainer":
        return cls([i for inst in insts for i in inst._datas])

    @classmethod
    @abc.abstractmethod
    def from_batch(cls, batch_data, batch_axis=0) -> "DataContainer":
        ...

    @classmethod
    @abc.abstractmethod
    def to_payload(cls, container: "DataContainer",) -> Payload:
        ...

    @classmethod
    @abc.abstractmethod
    def from_payload(cls, payload: Payload) -> "DataContainer":
        ...

    @classmethod
    @abc.abstractmethod
    def from_payloads(cls, payloads: t.List[Payload]) -> "DataContainer":
        pass


class NdarrayContainer(DataContainer["np.ndarray", "np.ndarray"]):
    def to_batch(self, batch_axis=0) -> "np.ndarray":
        import numpy as np

        return np.stack(self._datas, axis=batch_axis)

    @classmethod
    @abc.abstractmethod
    def from_batch(cls, batch_data, batch_axis=0) -> "DataContainer":
        import numpy as np

        return cls(np.split(batch_axis, batch_data.shape[batch_axis], axis=batch_axis))

    @classmethod
    @inject
    def to_payload(
        cls,
        container: "NdarrayContainer",
        plasma_db=Provide[BentoMLContainer.plasma_db],
    ) -> Payload:
        if plasma_db:
            return Payload(
                [plasma_db.put(d).binary() for d in container._datas], {"plasma": True},
            )

        return Payload(
            [cloudpickle.dumps(d) for d in container._datas], {"plasma": False},
        )

    @classmethod
    @inject
    def from_payload(
        cls, payload: Payload, plasma_db=Provide[BentoMLContainer.plasma_db]
    ):
        datas = payload.datas

        if payload.meta.get("plasma"):
            assert plasma_db

            import pyarrow.plasma as plasma

            return cls([plasma_db.get(plasma.ObjectID(i)) for i in datas])

        return cls([cloudpickle.loads(d) for d in datas])


class DataContainerRegistry:
    CONTAINER_SINGLE_TYPE_MAP: t.Dict[TypeRef, t.Type[DataContainer]] = dict()
    CONTAINER_BATCH_TYPE_MAP: t.Dict[TypeRef, t.Type[DataContainer]] = dict()

    @classmethod
    def register_container(
        cls,
        single_type: t.Union[TypeRef, type],
        batch_type: t.Union[TypeRef, type],
        container_cls: t.Type[DataContainer],
    ):
        single_type = TypeRef.from_type(single_type)
        batch_type = TypeRef.from_type(batch_type)

        cls.CONTAINER_BATCH_TYPE_MAP[batch_type] = container_cls
        cls.CONTAINER_SINGLE_TYPE_MAP[single_type] = container_cls

    @classmethod
    def find_by_single_type(cls, type_: type) -> t.Type["DataContainer"]:
        typeref = TypeRef.from_type(type_)
        return cls.CONTAINER_SINGLE_TYPE_MAP[typeref]

    @classmethod
    def find_by_batch_type(cls, type_: type) -> t.Type["DataContainer"]:
        typeref = TypeRef.from_type(type_)
        return cls.CONTAINER_BATCH_TYPE_MAP[typeref]


def register_builtin_containers():
    DataContainerRegistry.register_container(
        TypeRef("numpy", "ndarray"), TypeRef("numpy", "ndarray"), NdarrayContainer
    )
    # DataContainerRegistry.register_container(np.ndarray, np.ndarray, NdarrayContainer)


register_builtin_containers()


def single_data_to_container(single_data):
    container_cls = DataContainerRegistry.find_by_single_type(type(single_data))
    return container_cls([single_data])


def batch_data_to_container(batch_data, batch_axis=None):
    container_cls = DataContainerRegistry.find_by_batch_type(type(batch_data))
    container = container_cls.from_batch(batch_data, batch_axis=batch_axis)
    return container
