from __future__ import annotations

import typing as t
import logging
import os.path
import tempfile
from typing import TYPE_CHECKING

import pyarrow
from pyspark.files import SparkFiles
from pyspark.sql.dataframe import DataFrame

from .tag import Tag
from .bento import Bento
from ..bentos import get as get_bento
from ..bentos import import_bento
from .service import Service
from ..exceptions import NotFound
from ..exceptions import BentoMLException
from ..exceptions import MissingDependencyException
from .service.loader import load as load_service
from .service.loader import load_bento
from .service.loader import _load_bento as load_bento_service

try:
    import pyspark
    from pyspark.sql.functions import pandas_udf
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        '"pyspark" is required in order to use module `bentoml.spark`, install '
        "pyspark with `pip install pyspark`. For more information, refers to "
        "https://spark.apache.org/docs/latest/api/python/"
    )

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        '"pandas" is required in order to use module `bentoml.spark`, install '
        "pandas with `pip install pandas`. For more information, refers to "
        "https://pandas.pydata.org/"
    )


if TYPE_CHECKING:
    from pyspark.sql._typing import UserDefinedFunctionLike
    from pyspark.sql.session import SparkSession


logger = logging.getLogger(__name__)


def _distribute_bento(spark: SparkSession, bento: Bento) -> str:
    temp_dir = tempfile.mkdtemp()
    export_path = bento.export(temp_dir)
    spark.sparkContext.addFile(export_path)
    return os.path.basename(export_path)


def _load_bento(bento_tag: Tag):
    """
    load Bento from local bento store or the SparkFiles directory
    """
    try:
        return load_bento(str(bento_tag))
    except Exception:

        # Use the default Bento export file name. This relies on the implementation
        # of _distribute_bento to use default Bento export file name.
        bento_path = SparkFiles.get(f"{bento_tag.name}-{bento_tag.version}.bento")
        if not os.path.isfile(bento_path):
            raise

        import_bento(bento_path)
        return load_bento(str(bento_tag))


# SUPPORT_INPUT_IO_DESCRIPTORS = [
#     PandasDataFrame,
#     PandasSeries,
#     NumpyNdarray,
#     # Image,  # TODO
#     # File,  # TODO
#     Text,
# ]
# SUPPORT_OUTPUT_IO_DESCRIPTORS = [
#     PandasDataFrame,
#     PandasSeries,
#     NumpyNdarray,
#     # Image,  # TODO
#     # File,  # TODO
#     Text,
# ]


# PandasSeries
# In  -> pd.Series
# Out -> pd.Series
# Return Type -> PandasSeries.dtype (must be defined)
#
# NumpyNdarray
# In  -> pd.Series  # .to_numpy()
# Out -> pd.Series  # pd.Series(np_array)
# Return Type -> NumpyNdarray.dtype (must be defined)

# PandasDataframe
# In  -> list[pd.Series] | DataFrame
# Out -> DataFrame
# Return Type -> StructType
#
# Text
# In  -> pd.Series[str]
# Out -> pd.Series[str]
# Return Type -> Str


def _get_process(
    bento_tag: Tag, api_name: str
) -> t.Callable[[t.Iterator[pyarrow.RecordBatch]], t.Generator]:
    # def process(
    #     iterator: t.Iterator[api.input.spark_udf_input_type()]
    # ) -> t.Iterator[api.output.spark_udf_output_type()]:
    def process(
        iterator: t.Iterator[pyarrow.RecordBatch],
    ) -> t.Generator[
        pyarrow.RecordBatch, None, None
    ]:  # this type annotation is evaluated at runtime by
        # Initialize local service instance
        # TODO: support inference via remote bento server
        svc = _load_bento(bento_tag)
        for runner in svc.runners:
            runner.init_local(quiet=True)

        assert (
            api_name in svc.apis
        ), "An error occurred transferring the Bento to the Spark worker; see <something>."
        inference_api = svc.apis[api_name]
        assert inference_api.func is not None, "Inference API function not defined"

        for batch in iterator:
            # default batch size = 10,000
            func_input = inference_api.input.from_arrow(
                batch
            )  # use client api. call for input...?
            func_output = inference_api.func(
                func_input
            )  # instead of doing this, we'll start the server above
            # instead of running load bento, we just load bento
            # call the client here instead of doing the inference api.
            # the server does a bunch of smart things and we want to use it here...
            assert isinstance(func_output, pd.Series), f"type is {type(func_output)}"
            yield inference_api.output.to_arrow(func_output)

    return process  # type: ignore  # process type evaluated at runtime


def run_in_spark(
    spark: SparkSession, df: DataFrame, bento: Bento, api_name: str | None
) -> DataFrame:
    svc = load_bento_service(bento, False)

    if api_name is None:
        if len(svc.apis) != 1:
            raise BentoMLException(
                f'Bento "{bento.tag}" has multiple APIs ({svc.apis.keys()}), specify which API should be used for registering the UDF, e.g.: bentoml.spark.get_udf(spark, "my_service:latest", "predict")'
            )
        api_name = next(iter(svc.apis))
    else:
        if api_name not in svc.apis:
            raise BentoMLException(
                f"API name '{api_name}' not found in Bento '{bento.tag}', available APIs are {svc.apis.keys()}"
            )

    api = svc.apis[api_name]

    _distribute_bento(spark, bento)

    process = _get_process(bento.tag, api_name)

    if api.output._dtype is None:
        raise BentoMLException(
            f"Output descriptor for {api_name} must specify a dtype."
        )

    return df.mapInArrow(process, df.schema)  # TODO fix df.schema
