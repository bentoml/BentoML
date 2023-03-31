from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import bentoml._internal.external_typing as ext


def test_pep574_restore() -> None:
    import numpy as np

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
