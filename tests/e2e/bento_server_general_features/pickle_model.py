import typing as t

import numpy as np
import pandas as pd

from bentoml._internal.types import FileLike
from bentoml._internal.types import JSONSerializable


class PickleModel:
    def predict_file(self, input_files: t.List[FileLike[bytes]]) -> t.List[bytes]:
        return [f.read() for f in input_files]

    @classmethod
    def echo_json(cls, input_datas: JSONSerializable) -> JSONSerializable:
        return input_datas

    def echo_multi_ndarray(
        self,
        *input_arr: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> t.Tuple["np.ndarray[t.Any, np.dtype[t.Any]]", ...]:
        return input_arr

    def predict_ndarray(
        self,
        arr: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        assert isinstance(arr, np.ndarray)
        return arr * 2

    def predict_multi_ndarray(
        self,
        arr1: "np.ndarray[t.Any, np.dtype[t.Any]]",
        arr2: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        assert isinstance(arr1, np.ndarray)
        assert isinstance(arr2, np.ndarray)
        return (arr1 + arr2) // 2

    def predict_dataframe(self, df: "pd.DataFrame") -> "pd.DataFrame":
        assert isinstance(df, pd.DataFrame)
        output = df[["col1"]] * 2  # type: ignore
        assert isinstance(output, pd.DataFrame)
        return output
