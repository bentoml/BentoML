from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import bentoml._internal.external_typing as ext


def test_pep574_restore() -> None:
    import numpy as np
    import pandas as pd

    from bentoml._internal.utils.pickle import pep574_dumps
    from bentoml._internal.utils.pickle import pep574_loads

    arr1: ext.NpNDArray = np.random.uniform(size=(20, 20))
    arr2: ext.NpNDArray = np.random.uniform(size=(64, 64))
    arr3: ext.NpNDArray = np.random.uniform(size=(72, 72))

    lst = [arr1, arr2, arr3]

    bs: bytes
    concat_buffer_bs: bytes
    indices: list[int]
    bs, concat_buffer_bs, indices = pep574_dumps(lst)
    restored = t.cast(
        t.List["ext.NpNDArray"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for idx, arr in enumerate(lst):
        assert np.isclose(arr, restored[idx]).all()

    dic: dict[str, ext.NpNDArray] = dict(a=arr1, b=arr2, c=arr3)
    bs, concat_buffer_bs, indices = pep574_dumps(dic)
    restored = t.cast(
        t.Dict[str, "ext.NpNDArray"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for key, arr in dic.items():
        assert np.isclose(arr, restored[key]).all()

    df1: ext.PdDataFrame = pd.DataFrame(arr1)
    df2: ext.PdDataFrame = pd.DataFrame(arr2)
    df3: ext.PdDataFrame = pd.DataFrame(arr3)

    df_lst = [df1, df2, df3]

    bs, concat_buffer_bs, indices = pep574_dumps(df_lst)
    restored = t.cast(
        t.List["ext.PdDataFrame"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for idx, df in enumerate(df_lst):
        assert np.isclose(df.to_numpy(), restored[idx].to_numpy()).all()

    df_dic: dict[str, ext.PdDataFrame] = dict(a=df1, b=df2, c=df3)
    bs, concat_buffer_bs, indices = pep574_dumps(df_dic)
    restored = t.cast(
        t.Dict[str, "ext.PdDataFrame"], pep574_loads(bs, concat_buffer_bs, indices)
    )
    for key, df in df_dic.items():
        assert np.isclose(df.to_numpy(), restored[key].to_numpy()).all()
