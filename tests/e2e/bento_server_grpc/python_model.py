from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bentoml._internal.types import FileLike
    from bentoml._internal.types import JSONSerializable


class PythonFunction:
    def predict_file(self, files: list[FileLike[bytes]]) -> list[bytes]:
        return [f.read() for f in files]

    @classmethod
    def echo_json(cls, datas: JSONSerializable) -> JSONSerializable:
        return datas

    @classmethod
    def echo_ndarray(cls, datas: NDArray[Any]) -> NDArray[Any]:
        return datas

    def double_ndarray(self, data: NDArray[Any]) -> NDArray[Any]:
        assert isinstance(data, np.ndarray)
        return data * 2

    def multiply_float_ndarray(
        self, arr1: NDArray[np.float32], arr2: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        assert isinstance(arr1, np.ndarray)
        assert isinstance(arr2, np.ndarray)
        return arr1 * arr2

    def double_dataframe_column(self, df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame)
        return df[["col1"]] * 2  # type: ignore (no pandas types)

    def echo_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
