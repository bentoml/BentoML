from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import bentoml._internal.utils as utils
from bentoml._internal.types import LazyType
from bentoml._internal.types import MetadataDict


def test_typeref():

    # assert __eq__
    assert LazyType("numpy", "ndarray") == np.ndarray
    assert LazyType("numpy", "ndarray") == LazyType(type(np.array([2, 3])))

    # evaluate
    assert LazyType("numpy", "ndarray").get_class() == np.ndarray


def test_validate_labels():
    inp = {"label1": "label", "label3": "anotherlabel"}

    outp = inp.copy()
    utils.validate_labels(outp)

    assert inp == outp

    inp = {(12,): "non-string label key"}

    with pytest.raises(ValueError):
        utils.validate_labels(inp)  # type: ignore (testing bad types)

    inp = {"non-number label": 13}

    with pytest.raises(ValueError):
        utils.validate_labels(inp)  # type: ignore (testing bad types)

    inp = "non-dict labels"

    with pytest.raises(ValueError):
        utils.validate_labels(inp)  # type: ignore (testing bad types)


def test_validate_metadata():
    inp = "non-dict metadata"  # type: ignore (testing bad types)
    with pytest.raises(ValueError):
        utils.validate_metadata(inp)

    inp = {(12,): "non-string key"}  # type: ignore (testing bad types)
    with pytest.raises(ValueError):
        utils.validate_metadata(inp)

    # no validation required, inp == outp
    inp: MetadataDict = {
        "my key": 12,
        "float": 13.3,
        "string": "str",
        "date": datetime(2022, 3, 14),
        "timedelta": timedelta(days=3),
    }
    outp = inp.copy()
    utils.validate_metadata(outp)
    assert inp == outp

    inp: MetadataDict = {"ndarray": np.array([1, 2, 3])}  # type: ignore (we don't annotate translated types)
    expected = {"ndarray": [1, 2, 3]}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"uint": np.uint(3)}  # type: ignore (we don't annotate translated types)
    expected = {"uint": 3}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"date": np.datetime64("2022-03-17")}  # type: ignore (we don't annotate translated types)
    expected = {"date": date(2022, 3, 17)}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"spmatrix": csr_matrix([0, 0, 0, 0, 0, 1, 1, 0, 2])}  # type: ignore (we don't annotate translated types)
    with pytest.raises(ValueError):
        utils.validate_metadata(inp)

    inp: MetadataDict = {"series": pd.Series([1, 2, 4], name="myseriesname")}  # type: ignore (we don't annotate translated types)
    expected = {"series": {"myseriesname": {0: 1, 1: 2, 2: 4}}}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"pandasarray": pd.arrays.PandasArray(np.array([2, 4, 6]))}  # type: ignore (we don't annotate translated types)
    expected = {"pandasarray": [2, 4, 6]}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {
        "dataframe": pd.DataFrame(data={"col1": [1, 2], "col2": pd.Series({"a": 3, "b": 4})})  # type: ignore (we don't annotate translated types)
    }
    expected = {"dataframe": {"col1": {"a": 1, "b": 2}, "col2": {"a": 3, "b": 4}}}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"timestamp": pd.Timestamp(datetime(2022, 4, 12))}  # type: ignore (we don't annotate translated types)
    expected = {"timestamp": datetime(2022, 4, 12)}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"timedelta": pd.Timedelta(timedelta(2022))}  # type: ignore (we don't annotate translated types)
    expected = {"timedelta": timedelta(2022)}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"period": pd.Period("2012-05", freq="D")}  # type: ignore (we don't annotate translated types)
    expected = {"period": datetime(2012, 5, 1)}
    utils.validate_metadata(inp)
    assert inp == expected

    inp: MetadataDict = {"interval": pd.Interval(left=0, right=5)}  # type: ignore (we don't annotate translated types)
    expected = {"interval": (0, 5)}
    utils.validate_metadata(inp)
    assert inp == expected

    inp = {"unsupported": None}  # type: ignore (testing bad types)
    with pytest.raises(ValueError):
        utils.validate_metadata(inp)
