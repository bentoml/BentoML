from __future__ import annotations

import numpy as np
import pandas as pd

from bentoml.io import PandasDataFrame


def test_from_sample_serializes_dataframe_dtypes():
    dataframe = pd.DataFrame(
        {
            "int64": np.array([1, 2], dtype=np.int64),
            "float32": np.array([1.0, 2.0], dtype=np.float32),
            "bool": np.array([True, False], dtype=np.bool_),
            "nullable_int": pd.array([1, None], dtype="Int64"),
            "nullable_bool": pd.array([True, None], dtype="boolean"),
            "string": pd.array(["a", None], dtype="string"),
        }
    )

    descriptor = PandasDataFrame.from_sample(dataframe)

    assert descriptor.to_spec()["args"]["dtype"] == {
        "int64": "int64",
        "float32": "float32",
        "bool": "bool",
        "nullable_int": "Int64",
        "nullable_bool": "boolean",
        "string": "string",
    }


def test_from_sample_keeps_explicit_dtype_dict():
    descriptor = PandasDataFrame.from_sample(
        pd.DataFrame({"inferred": [1, 2]}),
        dtype={"configured": "string"},
    )

    assert descriptor.to_spec()["args"]["dtype"] == {"configured": "string"}
