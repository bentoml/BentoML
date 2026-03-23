from __future__ import annotations

import datetime
import os
import pathlib
import pickle
import uuid
from decimal import Decimal
from http import HTTPStatus

import pytest

from _bentoml_impl.safe_pickle import safe_pickle_loads
from _bentoml_impl.serde import Payload
from _bentoml_impl.serde import PickleSerde
from bentoml._internal.runner.container import Payload as RunnerPayload
from bentoml._internal.runner.utils import Params
from bentoml.exceptions import BentoMLException


def test_safe_pickle_loads_supports_selected_classes() -> None:
    expected = {
        "amount": Decimal("12.5"),
        "created_at": datetime.datetime(
            2026, 3, 23, 12, 0, tzinfo=datetime.timezone.utc
        ),
        "id": uuid.uuid4(),
        "path": pathlib.PurePosixPath("models/latest.bin"),
        "number": 42.125,
        "name": "Test Object",
    }

    payload = pickle.dumps(expected, protocol=5)

    actual = safe_pickle_loads(
        payload,
        allowed_classes=(
            Decimal,
            datetime.datetime,
            datetime.timezone,
            pathlib.PurePosixPath,
            uuid.UUID,
        ),
    )

    assert actual == expected


def test_safe_pickle_loads_rejects_unexpected_global() -> None:
    class Evil:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return os.system, ("echo 'EVIL'",)

    payload = pickle.dumps(Evil(), protocol=5)

    with pytest.raises(pickle.UnpicklingError, match="system"):
        safe_pickle_loads(payload)


def test_safe_pickle_loads_supports_runner_params_payload() -> None:
    expected = Params(
        RunnerPayload(
            data=b"payload",
            meta={"format": "default"},
            container="NdarrayContainer",
            batch_size=1,
        ),
        item=RunnerPayload(
            data=b"payload-2",
            meta={"format": "default"},
            container="NdarrayContainer",
            batch_size=2,
        ),
    )

    actual = safe_pickle_loads(
        pickle.dumps(expected, protocol=5),
        allowed_classes=(Params, RunnerPayload),
    )

    assert actual.args == expected.args
    assert actual.kwargs == expected.kwargs


def test_pickle_serde_rejects_unsafe_payload() -> None:
    class Evil:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return os.system, ("echo 'EVIL'",)

    serde = PickleSerde()

    with pytest.raises(BentoMLException) as exc_info:
        serde.deserialize_value(Payload((pickle.dumps(Evil(), protocol=5),)))

    assert exc_info.value.error_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_pickle_serde_round_trips_numpy_array() -> None:
    np = pytest.importorskip("numpy")
    serde = PickleSerde()
    array = np.arange(6).reshape(2, 3)

    payload = serde.serialize_value(array)
    actual = serde.deserialize_value(payload)

    assert np.array_equal(actual, array)


def test_pickle_serde_round_trips_pandas_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    serde = PickleSerde()
    dataframe = pd.DataFrame({"left": [1, 2], "right": [3, 4]})

    payload = serde.serialize_value(dataframe)
    actual = serde.deserialize_value(payload)

    assert actual.equals(dataframe)


def test_pickle_serde_round_trips_pil_image() -> None:
    pil_image = pytest.importorskip("PIL.Image")
    serde = PickleSerde()
    image = pil_image.new("RGB", (2, 2), color="red")

    payload = serde.serialize_value(image)
    actual = serde.deserialize_value(payload)

    assert actual.mode == image.mode
    assert actual.size == image.size
    assert actual.tobytes() == image.tobytes()


def test_pickle_serde_round_trips_torch_tensor() -> None:
    torch = pytest.importorskip("torch")
    serde = PickleSerde()
    tensor = torch.arange(6).reshape(2, 3)

    payload = serde.serialize_value(tensor)
    actual = serde.deserialize_value(payload)

    assert torch.equal(actual, tensor)


def test_pickle_serde_round_trips_tensorflow_tensor() -> None:
    tf = pytest.importorskip("tensorflow")
    serde = PickleSerde()
    tensor = tf.constant(7)

    payload = serde.serialize_value(tensor)
    actual = serde.deserialize_value(payload)

    assert bool(tf.reduce_all(tf.equal(actual, tensor)))
