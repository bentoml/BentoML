import numpy as np
import pandas as pd
import pytest

import bentoml._internal.runner.container as c


@pytest.mark.parametrize("batch_axis_exc", [AssertionError])
@pytest.mark.parametrize("wrong_batch_axis", [1, 19])
def test_default_container(batch_axis_exc, wrong_batch_axis):

    _list = [1, 2, 3]
    assert c.DefaultContainer.singles_to_batch(_list) == _list
    assert c.DefaultContainer.batch_to_singles(_list) == _list

    def _generator():
        yield "apple"
        yield "banana"
        yield "cherry"

    assert c.DefaultContainer.payload_to_single(
        c.DefaultContainer.single_to_payload(_generator())
    ) == list(_generator())
    assert c.DefaultContainer.payload_to_batch(
        c.DefaultContainer.batch_to_payload(_generator())
    ) == list(_generator())

    # DefaultContainer should only allow batch_axis = 0
    with pytest.raises(batch_axis_exc):
        c.DefaultContainer.singles_to_batch(_list, batch_axis=wrong_batch_axis)

    with pytest.raises(batch_axis_exc):
        c.DefaultContainer.batch_to_singles(_list, batch_axis=wrong_batch_axis)


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_ndarray_container(batch_axis):

    single_array = np.arange(6).reshape(2, 3)
    singles = [single_array, single_array]
    batch_array = np.stack(singles, axis=batch_axis)

    assert (
        c.NdarrayContainer.singles_to_batch(singles, batch_axis=batch_axis)
        == batch_array
    ).all()
    assert (
        c.NdarrayContainer.batch_to_singles(batch_array, batch_axis=batch_axis)[0]
        == single_array
    ).all()


@pytest.mark.parametrize("batch_axis_exc", [AssertionError])
@pytest.mark.parametrize("wrong_batch_axis", [1, 19])
def test_pandas_container(batch_axis_exc, wrong_batch_axis):

    d = {"a": 1, "b": 2, "c": 3}
    ser = pd.Series(data=d, index=["a", "b", "c"])
    ser_singles = [ser, ser]
    single_df = ser.to_frame().T
    df_singles = [single_df, single_df]

    batch_df = c.PandasDataFrameContainer.singles_to_batch(ser_singles)

    ser_batch_df = c.PandasDataFrameContainer.singles_to_batch(ser_singles)
    assert batch_df.equals(ser_batch_df)

    # df_batch_df = c.PandasDataFrameContainer.singles_to_batch(df_singles)
    # assert batch_df.equals(df_batch_df)

    # PandasDataFrameContainer should only allow batch_axis = 0

    with pytest.raises(batch_axis_exc):
        c.PandasDataFrameContainer.singles_to_batch(
            df_singles, batch_axis=wrong_batch_axis
        )

    with pytest.raises(batch_axis_exc):
        c.PandasDataFrameContainer.batch_to_singles(
            batch_df, batch_axis=wrong_batch_axis
        )
