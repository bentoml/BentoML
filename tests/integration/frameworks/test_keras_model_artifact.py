import os
import typing as t

import numpy as np
import pytest
import tensorflow.keras as tfk

from bentoml.keras import KerasModel

test_data: t.List[int] = [1, 2, 3, 4, 5]


@pytest.fixture(scope='session')
def keras_model() -> "tfk.models.Model":
    net = tfk.Sequential(
        (
            tfk.layers.Dense(
                units=1,
                input_shape=(5,),
                use_bias=False,
                kernel_initializer=tfk.initializers.Ones(),
            ),
        )
    )
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return net


@pytest.mark.parametrize("kwargs", [{"store_as_json": True}, {"store_as_json": False},])
def test_keras_save_load(kwargs, keras_model, tmpdir):

    KerasModel(keras_model, **kwargs).save(tmpdir)
    if kwargs['store_as_json']:
        assert os.path.exists(KerasModel.get_path(tmpdir, '_json.json'))
        assert os.path.exists(KerasModel.get_path(tmpdir, '_weights.hdf5'))
    else:
        assert os.path.exists(KerasModel.get_path(tmpdir, '.h5'))

    keras_loaded: "tfk.models.Model" = KerasModel.load(tmpdir)
    assert keras_loaded.predict(np.array([test_data])) == keras_model.predict(
        np.array([test_data])
    )
