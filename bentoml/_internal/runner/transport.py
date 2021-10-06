import abc
from typing import TYPE_CHECKING, Generic, Iterator, Sequence, TypeVar, Union, overload

from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer

SingleType = TypeVar("SingleType")
BatchType = TypeVar("BatchType")
PayloadType = TypeVar("PayloadType")

IndexType = Union[None, int]

if TYPE_CHECKING:
    import numpy as np


class Container(Generic[SingleType, BatchType, PayloadType]):
    @abc.abstractmethod
    def put_one(self, data: SingleType) -> None:
        ...

    @abc.abstractmethod
    def put_batch(self, batch_data: BatchType) -> IndexType:
        ...

    @overload
    @abc.abstractmethod
    def iter(self, indexes: None = None) -> Iterator[SingleType]:
        ...

    @overload
    @abc.abstractmethod
    def iter(
        self,
        indexes: Sequence[IndexType],
    ) -> Iterator[Union[SingleType, BatchType]]:
        ...

    @abc.abstractmethod
    def iter(
        self, indexes: Union[Sequence[IndexType], None] = None
    ) -> Union[Iterator[Union[SingleType, BatchType]], Iterator[SingleType]]:
        ...

    @abc.abstractmethod
    def squeeze(self) -> BatchType:
        ...

    @abc.abstractmethod
    def to_payload(self) -> PayloadType:
        ...

    @abc.abstractmethod
    def from_payload(self, payload: PayloadType) -> "Container":
        ...


class NdarrayContainer(Container["np.ndarray", "np.ndarray", bytes]):
    def __init__(self, batch_axis=0):
        self.batch_axis = batch_axis
        self._datas = []
        self.indexes = []

    def put_one(self, data):
        self.indexes.append(None)
        self._datas.append(data)

    def put_batch(self, batch_data):
        batch_size = batch_data.shape[self.batch_axis]
        self.indexes.append(batch_size)
        self._datas.append(batch_data)
        return batch_size

    def iter(self, indexes=None):
        if indexes is None:
            for d, i in zip(self._datas, self.indexes):
                if i is None:
                    yield d
                else:
                    leading_indices = (slice(None),) * self.batch_axis
                    for j in range(d.shape[self.batch_axis]):
                        yield d[leading_indices + (j,)]
        # TODO(jiang): when indexes exsist

    def squeeze(self):
        pass

    @inject
    def _get_plasma(self, plasma=Provide[BentoMLContainer.plasma_db]):
        return plasma

    def to_payload(self, input_):
        plasma = self._get_plasma()
        if plasma:
            return plasma.put(input_)
        # TODO(jiang): when plasma not exsist

    def from_payload(self, payload):
        plasma = self._get_plasma()
        if plasma:
            return plasma.get(payload)
        # TODO(jiang): when plasma not exsist


TYPE_SINGLE_MAPPING = {"np.ndarray": NdarrayContainer}

TYPE_BATCH_MAPPING = {"np.ndarray": NdarrayContainer}


def get_container_cls_by_single(type_):
    return TYPE_SINGLE_MAPPING[type_]


def get_container_cls_by_batch(type_):
    return TYPE_BATCH_MAPPING[type_]


def register_container(single_type, batch_type, container_cls):
    TYPE_BATCH_MAPPING[batch_type] = container_cls
    TYPE_SINGLE_MAPPING[single_type] = container_cls
