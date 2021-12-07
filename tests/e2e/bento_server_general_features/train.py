import typing as t

import numpy as np
import pandas as pd

import bentoml
import bentoml.sklearn
from bentoml._internal.types import FileLike
from bentoml._internal.types import JSONSerializable


class PickleModel:
    @staticmethod
    def predict_file(input_files: t.List[FileLike]) -> t.List[bytes]:
        return [f.read() for f in input_files]

    @staticmethod
    def echo_json(input_datas: JSONSerializable) -> JSONSerializable:
        return input_datas

    @staticmethod
    def echo_multi_ndarray(
        *input_arr: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> t.Tuple["np.ndarray[t.Any, np.dtype[t.Any]]", ...]:
        return input_arr

    @staticmethod
    def predict_ndarray(
        arr: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        assert isinstance(arr, np.ndarray)
        return arr * 2

    @staticmethod
    def predict_multi_ndarray(
        arr1: "np.ndarray[t.Any, np.dtype[t.Any]]",
        arr2: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        assert isinstance(arr1, np.ndarray)
        assert isinstance(arr2, np.ndarray)
        return (arr1 + arr2) // 2

    @staticmethod
    def predict_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
        assert isinstance(df, pd.DataFrame)
        output = df[["col1"]] * 2
        assert isinstance(output, pd.DataFrame)
        return output


def train():
    bentoml.sklearn.save("sk_model", PickleModel())


if __name__ == "__main__":
    train()
