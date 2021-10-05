from typing import Generic, Iterator, List, Optional, Sequence, TypeVar, Union

DataType = TypeVar("DataType")
BatchType = TypeVar("BatchType")


IndexType = Union[int, slice]


import numpy as np


class BatchContainer(Generic[DataType, BatchType]):
    TYPE_MAPPING = {}

    def __init__(self, **options):
        self.options = options
        self._is_seqled = False
        self._datas = []
        self._indexes = []

    def __len__(self):
        assert self._is_seqled
        return sum((1 if i is None else i for i in self._indexes))

    def flatten(self) -> Iterator[DataType]:
        raise NotImplementedError()

    def squeese(self) -> BatchType:
        raise NotImplementedError()

    @classmethod
    def register(cls, type_, container_cls):
        cls.TYPE_MAPPING[type_] = container_cls

    def unbox(self, indexes):
        assert self._is_seqled

    def box(self, data_list, indexes: Optional[Sequence[IndexType]] = None):
        pass


class NdarrayContainer(BatchContainer[np.ndarray, np.ndarray]):
    def __init__(self, batch_axis=0):
        super().__init__(batch_axis=batch_axis)

    def squeese(self):
        if all(i is None for i in self._indexes):
            return np.stack(self._datas)
        batches = tuple(
            np.expand_dims(a, axis=self.options['batch_axis']) if i is None else a
            for a, i in zip(self._datas, self._indexes)
        )
        return np.concatenate(batches)

    def flatten(self):
        for data, index in zip(self._datas, self._indexes):
            if index is None:
                yield data
            else:
                for i in range(index):
                    yield data[i]

    def box(self, data_list, indexes: Optional[Sequence[IndexType]] = None):
        pass


class NdarrayBox(BatchContainer[np.ndarray, np.ndarray]):
    def __init__(self, raw, size, , **options):
        pass

    def unbox(self, indexes: Sequence[IndexType]) -> List[np.ndarray]:
        assert sum(1 if i is None else i)


box = NdarrayBox(a)


BatchContainer.register(np.ndarray, NdarrayContainer)
