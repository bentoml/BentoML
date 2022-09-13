import fsspec
import typing as t
from bentoml._internal.io_descriptors.pandas import SerializationFormat
from bentoml.io import (
    File,
    JSON,
    Text,
    Image,
    NumpyNdarray,
    PandasSeries,
    PandasDataFrame,
    Multipart,
)

bento_all_io = (
    File,
    JSON,
    Text,
    Image,
    NumpyNdarray,
    PandasSeries,
    PandasDataFrame,
    Multipart,
)


def is_equal(
    io_desc: t.Union[bento_all_io], expected_obj: t.Any, actual_obj: t.Any
) -> bool:
    if isinstance(io_desc, NumpyNdarray):
        return (expected_obj == actual_obj).all()
    elif isinstance(io_desc, PandasDataFrame):
        return expected_obj.equals(actual_obj)
    else:
        raise NotImplementedError(f"IO descriptor {io_desc} is not supported")


def get_test_data(io_desc: t.Union[bento_all_io], io_string: str) -> t.Any:
    io_desc_to_use = io_desc

    # TODO: find a better way to check if its a file. remember io_string can be a local file path, cloud file path (s3, gcs, etc), or an actual data (pandas df, numpy array, text, etc)
    if "/" in io_string:
        with fsspec.open(io_string, mode="rt") as f:
            data = f.read()
        if isinstance(io_desc, PandasDataFrame):
            file_ext = io_string.split(".")[-1]
            file_ext = file_ext.upper()
            try:
                SerializationFormat[file_ext]
            except KeyError:
                raise ValueError(f'File extension "{file_ext}" is not supported')
            io_desc_to_use = PandasDataFrame(default_format=file_ext)

    else:
        data = io_string

    return io_desc_to_use.deserialize(data)
