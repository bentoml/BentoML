

import ctypes
from typing import Any

from .core import DataIter, DMatrix, _ProxyDMatrix

'''Data dispatching for DMatrix.'''
c_bst_ulong = ctypes.c_uint64
_pandas_dtype_mapper = ...
_dt_type_mapper = ...
_dt_type_mapper2 = ...
def dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical=...):
    '''Dispatch data for DMatrix.'''
    ...

def dispatch_meta_backend(matrix: DMatrix, data, name: str, dtype: str = ...): # -> None:
    '''Dispatch for meta info.'''
    ...

class SingleBatchInternalIter(DataIter):
    '''An iterator for single batch data to help creating device DMatrix.
    Transforming input directly to histogram with normal single batch data API
    can not access weight for sketching.  So this iterator acts as a staging
    area for meta info.

    '''
    def __init__(self, data, label, weight, base_margin, group, qid, label_lower_bound, label_upper_bound, feature_weights, feature_names, feature_types) -> None:
        ...
    
    def next(self, input_data): # -> Literal[0, 1]:
        ...
    
    def reset(self): # -> None:
        ...
    


def dispatch_device_quantile_dmatrix_set_data(proxy: _ProxyDMatrix, data: Any) -> None:
    '''Dispatch for DeviceQuantileDMatrix.'''
    ...

