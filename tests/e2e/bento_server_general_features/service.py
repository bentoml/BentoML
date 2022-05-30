import typing as t

import numpy as np
import pandas as pd
import pydantic
from PIL.Image import Image as PILImage
from PIL.Image import fromarray

import bentoml
import bentoml.picklable_model
from bentoml.io import File
from bentoml.io import JSON
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.io import NumpyNdarray
from bentoml.io import PandasDataFrame
from bentoml._internal.types import FileLike
from bentoml._internal.types import JSONSerializable

py_model = bentoml.picklable_model.get("py_model").to_runner()


svc = bentoml.Service(
    name="general",
    runners=[py_model],
)


@svc.api(input=JSON(), output=JSON())
async def echo_json(json_obj: JSONSerializable) -> JSONSerializable:
    batch_ret = await py_model.echo_json.async_run([json_obj])
    return batch_ret[0]


@svc.api(input=JSON(), output=JSON())
def echo_json_sync(json_obj: JSONSerializable) -> JSONSerializable:
    batch_ret = py_model.echo_json.run([json_obj])
    return batch_ret[0]


class _Schema(pydantic.BaseModel):
    name: str
    endpoints: t.List[str]


@svc.api(
    input=JSON(pydantic_model=_Schema),
    output=JSON(),
)
async def pydantic_json(json_obj: JSONSerializable) -> JSONSerializable:
    batch_ret = await py_model.echo_json.async_run([json_obj])
    return batch_ret[0]


@svc.api(
    input=NumpyNdarray(shape=(2, 2), enforce_shape=True),
    output=NumpyNdarray(shape=(2, 2)),
)
async def predict_ndarray_enforce_shape(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.shape == (2, 2)
    return await py_model.predict_ndarray.async_run(inp)


@svc.api(
    input=NumpyNdarray(dtype="uint8", enforce_dtype=True),
    output=NumpyNdarray(dtype="str"),
)
async def predict_ndarray_enforce_dtype(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.dtype == np.dtype("uint8")
    return await py_model.predict_ndarray.async_run(inp)


@svc.api(
    input=PandasDataFrame(dtype={"col1": "int64"}, orient="records"),
    output=PandasDataFrame(),
)
async def predict_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    assert df["col1"].dtype == "int64"
    output = await py_model.predict_dataframe.async_run(df)
    dfo = pd.DataFrame()
    dfo["col1"] = output
    assert isinstance(dfo, pd.DataFrame)
    return dfo


@svc.api(input=File(), output=File())
async def predict_file(f: FileLike[bytes]) -> bytes:
    batch_ret = await py_model.predict_file.async_run([f])
    return batch_ret[0]


@svc.api(input=Image(), output=Image(mime_type="image/bmp"))
async def echo_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    return np.array(f)  # type: ignore[arg-type]


@svc.api(
    input=Multipart(original=Image(), compared=Image()),
    output=Multipart(img1=Image(), img2=Image()),
)
async def predict_multi_images(
    original: t.Dict[str, Image], compared: t.Dict[str, Image]
):
    output_array = await py_model.predict_multi_ndarray.async_run(
        np.array(original), np.array(compared)
    )
    img = fromarray(output_array)
    return dict(img1=img, img2=img)
