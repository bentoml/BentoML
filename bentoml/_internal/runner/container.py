import abc
import typing as t
from typing import TYPE_CHECKING, Generic, Iterator, Sequence, TypeVar, Union

from .utils import TypeRef

SingleType = TypeVar("SingleType")
BatchType = TypeVar("BatchType")
PayloadType = TypeVar("PayloadType")

IndexType = Union[None, int]

if TYPE_CHECKING:
    import numpy as np


class Container(Generic[SingleType, BatchType, PayloadType]):  # TODO(jiang): naming
    def __init__(self, batch_axis=None) -> None:
        self.batch_axis = batch_axis

    @abc.abstractmethod
    def put_single(self, data: SingleType) -> None:
        ...

    @abc.abstractmethod
    def put_batch(self, batch_data: BatchType) -> IndexType:
        ...

    @abc.abstractmethod
    def slice_single(self) -> Iterator[SingleType]:
        ...

    @abc.abstractmethod
    def slice(
        self, indexes: Sequence[IndexType]
    ) -> Iterator[Union[SingleType, BatchType]]:
        ...

    @abc.abstractmethod
    def squeeze(self) -> BatchType:
        ...


class NdarrayContainer(Container["np.ndarray", "np.ndarray", bytes]):
    def __init__(self, batch_axis=None) -> None:
        super().__init__(batch_axis=batch_axis)
        self._datas = []
        self.indexes = []

    def put_single(self, data):
        self.indexes.append(None)
        self._datas.append(data)

    def put_batch(self, batch_data):
        batch_size = batch_data.shape[self.batch_axis]
        self.indexes.append(batch_size)
        self._datas.append(batch_data)
        return batch_size

    def slice_single(self) -> Iterator["np.ndarray"]:
        for d, i in zip(self._datas, self.indexes):
            if i is None:
                yield d
            else:
                leading_indices = (slice(None),) * self.batch_axis
                for j in range(d.shape[self.batch_axis]):
                    yield d[leading_indices + (j,)]

    def slice(self, indexes) -> Iterator["np.ndarray"]:
        squeezed_batch = self.squeeze()

        cursor = 0
        leading_indices = (slice(None),) * self.batch_axis  # to slice
        for i in indexes:
            if i is None:
                yield squeezed_batch[leading_indices + (cursor,)]
                cursor += 1
            else:
                yield squeezed_batch[leading_indices + (slice(cursor, cursor + i - 1),)]
                cursor += i

        assert (
            squeezed_batch.shape[self.batch_axis] == cursor
        ), "indexes did not match the length of the batch"

    def squeeze(self) -> "np.ndarray":
        import numpy as np

        if all(i is None for i in self.indexes):
            return np.stack(self._datas, axis=self.batch_axis)

        batch_datas = tuple(
            d if i is not None else np.expand_dims(d, axis=self.batch_axis)
            for d, i in zip(self._datas, self.indexes)
        )
        return np.concatenate(batch_datas, axis=self.batch_axis)


class DataContainerRegistry:
    CONTAINER_SINGLE_TYPE_MAP: t.Dict[TypeRef, t.Type[Container]] = dict()
    CONTAINER_BATCH_TYPE_MAP: t.Dict[TypeRef, t.Type[Container]] = dict()

    @classmethod
    def register_container(
        cls,
        single_type: t.Union[TypeRef, type],
        batch_type: t.Union[TypeRef, type],
        container_cls: t.Type[Container],
    ):
        single_type = TypeRef.from_type(single_type)
        batch_type = TypeRef.from_type(batch_type)

        cls.CONTAINER_BATCH_TYPE_MAP[batch_type] = container_cls
        cls.CONTAINER_SINGLE_TYPE_MAP[single_type] = container_cls

    @classmethod
    def find_container_by_single_type(cls, type_: type) -> t.Type["Container"]:
        typeref = TypeRef.from_type(type_)
        return cls.CONTAINER_SINGLE_TYPE_MAP[typeref]

    @classmethod
    def find_container_by_batch_type(cls, type_: type) -> t.Type["Container"]:
        typeref = TypeRef.from_type(type_)
        return cls.CONTAINER_BATCH_TYPE_MAP[typeref]


def register_builtin_containers():
    DataContainerRegistry.register_container(
        TypeRef("numpy", "ndarray"), TypeRef("numpy", "ndarray"), NdarrayContainer
    )
    # DataContainerRegistry.register_container(np.ndarray, np.ndarray, NdarrayContainer)


register_builtin_containers()


def single_data_to_container(single_data, batch_axis=None):
    container_cls = DataContainerRegistry.find_container_by_single_type(
        type(single_data)
    )
    container = container_cls(batch_axis=batch_axis)
    container.put_single(single_data)
    return container


def batch_data_to_container(batch_data, batch_axis=None):
    container_cls = DataContainerRegistry.find_container_by_batch_type(type(batch_data))
    container = container_cls(batch_axis=batch_axis)
    container.put_batch(batch_data)
    return container
