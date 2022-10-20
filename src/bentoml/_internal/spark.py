from __future__ import annotations

import typing as t
import logging
import os.path
import tempfile
from typing import TYPE_CHECKING

from ..bentos import import_bento
from ..bentos import get as get_bento
from .service.loader import load as load_service
from .bento import Bento
from ..exceptions import NotFound
from ..exceptions import BentoMLException
from ..exceptions import MissingDependencyException
from .tag import Tag
from .service.loader import load_bento
from .service import Service

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


def _load_bento(bento_tag: Tag) -> Service:
    """
    load Bento from local bento store or the SparkFiles directory
    """
    try:
        return load_bento(str(bento_tag))
    except NotFound:
        from pyspark.files import SparkFiles

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


def get_udf(
    spark: SparkSession, bento_tag: Tag | str, api_name: str | None
) -> UserDefinedFunctionLike:
    """
    Example Usage:

    bento_udf = bentoml.spark.get_udf(spark, "iris_classifier:latest", "predict", )

    Args:
        spark: the Spark session for registering the UDF
        bento_tag: target Bento to run, the tag must be found in driver's local Bento store
        api_name: specify which API to run, a Bento Service may contain multiple APIs

    Returns:
        A pandas_udf for running target Bento on Spark DataFrame
    """
    svc = load_service(str(bento_tag))
    assert svc.tag is not None
    bento_tag = svc.tag  # resolved tag, no more "latest" here

    if api_name is None:
        if len(svc.apis) != 1:
            raise BentoMLException(
                f'Bento "{bento_tag}" has multiple APIs ({svc.apis.keys()}), specify which API should be used for registering the UDF, e.g.: bentoml.spark.get_udf(spark, "my_service:latest", "predict")'
            )
        api_name = next(iter(svc.apis))
    else:
        if api_name not in svc.apis:
            raise BentoMLException(
                f"API name '{api_name}' not found in Bento '{bento_tag}', available APIs are {svc.apis.keys()}"
            )

    api = svc.apis[api_name]

    # Validate API io descriptors are supported for Spark UDF
    # if api.input.__class__ not in SUPPORT_INPUT_IO_DESCRIPTORS:
    #     raise BentoMLException(f"Service API input type {api.input.__class__} is not supported for Spark UDF conversion")
    # if api.output.__class__ not in SUPPORT_OUTPUT_IO_DESCRIPTORS:
    #     raise BentoMLException(f"Service API output type {api.output.__class__} is not supported for Spark UDF conversion")

    # Distribute Bento file to worker nodes
    bento_filename = _distribute_bento(spark, get_bento(bento_tag))

    # def process(
    #     iterator: t.Iterator[api.input.spark_udf_input_type()]
    # ) -> t.Iterator[api.output.spark_udf_output_type()]:
    def process(iterator: t.Iterator[tuple[pd.Series[t.Any]]]) -> t.Iterator[tuple[pd.Series[t.Any]]]:
        # Initialize local service instance
        # TODO: support inference via remote bento server
        svc = _load_bento(bento_tag)
        for runner in svc.runners:
            runner.init_local(quiet=True)

        assert (
            api_name in svc.apis
        ), "An error occurred transferring the Bento to the Spark worker; see <something>."
        inference_api = svc.apis[api_name]

        for input_batch in iterator:
            func_input = inference_api.input.to_pandas_series(input_batch)
            func_output = inference_api.func(func_input)
            yield inference_api.output.from_pandas_series(func_output)

        for input_args in iterator:
            # default batch size = 10,000

            yield api.output.to_pandas_series(inference_api.func(api.input.from_pandas_series(input_args)))

    return pandas_udf(process)
