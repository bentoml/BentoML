from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bentoml._internal.types import FileLike
    from bentoml._internal.types import JSONSerializable


class PickleModel:
    def predict_file(self, input_files: t.List[FileLike[bytes]]) -> t.List[bytes]:
        return [f.read() for f in input_files]

    @classmethod
    def echo_json(cls, input_datas: JSONSerializable) -> JSONSerializable:
        return input_datas

    @classmethod
    def echo_obj(cls, input_datas: t.Any) -> t.Any:
        return input_datas

    def echo_multi_ndarray(self, *input_arr: NDArray[t.Any]) -> tuple[NDArray[t.Any]]:
        return input_arr

    def predict_ndarray(
        self,
        arr: NDArray[t.Any],
        coefficient: int = 1,
    ) -> NDArray[t.Any]:
        assert isinstance(arr, np.ndarray)
        return arr * coefficient

    def predict_multi_ndarray(
        self, arr1: NDArray[t.Any], arr2: NDArray[t.Any]
    ) -> NDArray[t.Any]:
        assert isinstance(arr1, np.ndarray)
        assert isinstance(arr2, np.ndarray)
        return (arr1 + arr2) // 2

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame)
        output = df[["col1"]] * 2  # type: ignore
        assert isinstance(output, pd.DataFrame)
        return output
