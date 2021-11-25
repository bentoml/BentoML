

from typing import Hashable, Literal

import numpy as np

def unique_label_indices(labels: np.ndarray) -> np.ndarray:
    ...

class Factorizer:
    count: int
    def __init__(self, size_hint: int) -> None:
        ...
    
    def get_count(self) -> int:
        ...
    


class ObjectFactorizer(Factorizer):
    table: PyObjectHashTable
    uniques: ObjectVector
    def factorize(self, values: np.ndarray, sort: bool = ..., na_sentinel=..., na_value=...) -> np.ndarray:
        ...
    


class Int64Factorizer(Factorizer):
    table: Int64HashTable
    uniques: Int64Vector
    def factorize(self, values: np.ndarray, sort: bool = ..., na_sentinel=..., na_value=...) -> np.ndarray:
        ...
    


class Int64Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class Int32Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class Int16Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class Int8Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class UInt64Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class UInt32Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class UInt16Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class UInt8Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class Float64Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class Float32Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class Complex128Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class Complex64Vector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class StringVector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class ObjectVector:
    def __init__(self) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def to_array(self) -> np.ndarray:
        ...
    


class HashTable:
    def __init__(self, size_hint: int = ...) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __contains__(self, key: Hashable) -> bool:
        ...
    
    def sizeof(self, deep: bool = ...) -> int:
        ...
    
    def get_state(self) -> dict[str, int]:
        ...
    
    def get_item(self, item):
        ...
    
    def set_item(self, item) -> None:
        ...
    
    def map(self, keys: np.ndarray, values: np.ndarray) -> None:
        ...
    
    def map_locations(self, values: np.ndarray) -> None:
        ...
    
    def lookup(self, values: np.ndarray) -> np.ndarray:
        ...
    
    def get_labels(self, values: np.ndarray, uniques, count_prior: int = ..., na_sentinel: int = ..., na_value: object = ...) -> np.ndarray:
        ...
    
    def unique(self, values: np.ndarray, return_inverse: bool = ...) -> tuple[np.ndarray, np.ndarray], | np.ndarray:
        ...
    
    def factorize(self, values: np.ndarray, na_sentinel: int = ..., na_value: object = ..., mask=...) -> tuple[np.ndarray, np.ndarray],:
        ...
    


class Complex128HashTable(HashTable):
    ...


class Complex64HashTable(HashTable):
    ...


class Float64HashTable(HashTable):
    ...


class Float32HashTable(HashTable):
    ...


class Int64HashTable(HashTable):
    def get_labels_groupby(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray],:
        ...
    


class Int32HashTable(HashTable):
    ...


class Int16HashTable(HashTable):
    ...


class Int8HashTable(HashTable):
    ...


class UInt64HashTable(HashTable):
    ...


class UInt32HashTable(HashTable):
    ...


class UInt16HashTable(HashTable):
    ...


class UInt8HashTable(HashTable):
    ...


class StringHashTable(HashTable):
    ...


class PyObjectHashTable(HashTable):
    ...


def duplicated_int64(values: np.ndarray, keep: Literal["last", "first", False] = ...) -> np.ndarray:
    ...

def mode_int64(values: np.ndarray, dropna: bool) -> np.ndarray:
    ...

def value_count_int64(values: np.ndarray, dropna: bool) -> tuple[np.ndarray, np.ndarray],:
    ...

def duplicated(values: np.ndarray, keep: Literal["last", "first", False] = ...) -> np.ndarray:
    ...

def mode(values: np.ndarray, dropna: bool) -> np.ndarray:
    ...

def value_count(values: np.ndarray, dropna: bool) -> tuple[np.ndarray, np.ndarray],:
    ...

def ismember(arr: np.ndarray, values: np.ndarray) -> np.ndarray:
    ...

def object_hash(obj) -> int:
    ...

def objects_are_equal(a, b) -> bool:
    ...

