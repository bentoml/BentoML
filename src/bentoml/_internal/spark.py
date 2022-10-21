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


def _load_bento(bento_tag: Tag):
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

def _get_process(bento_tag: Tag, api_name: str) -> t.Callable[[t.Iterator[tuple[pd.Series[t.Any]]]], t.Iterator[pd.Series[t.Any]]]:
    # def process(
    #     iterator: t.Iterator[api.input.spark_udf_input_type()]
    # ) -> t.Iterator[api.output.spark_udf_output_type()]:
    def process(iterator: t.Iterator[t.Tuple[pd.Series]]) -> t.Iterator[pd.Series]:  # type: ignore  # this type annotation is evaluated at runtime by 
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

        for input_args in t.cast("t.Iterator[tuple[pd.Series[t.Any]]]", iterator):
            # default batch size = 10,000
            func_input = inference_api.input.from_pandas_series(input_args)
            func_output = inference_api.func(func_input)
            assert isinstance(func_output, pd.Series), f"type is {type(func_output)}"
            yield inference_api.output.to_pandas_series(func_output)

    return process  # type: ignore  # process type evaluated at runtime
    
    
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
    svc2 = load_service(str(bento_tag))
    assert svc2.tag is not None
    bento_tag = svc2.tag  # resolved tag, no more "latest" here

    if api_name is None:
        if len(svc2.apis) != 1:
            raise BentoMLException(
                f'Bento "{bento_tag}" has multiple APIs ({svc2.apis.keys()}), specify which API should be used for registering the UDF, e.g.: bentoml.spark.get_udf(spark, "my_service:latest", "predict")'
            )
        api_name = next(iter(svc2.apis))
    else:
        if api_name not in svc2.apis:
            raise BentoMLException(
                f"API name '{api_name}' not found in Bento '{bento_tag}', available APIs are {svc2.apis.keys()}"
            )

    api = svc2.apis[api_name]

    # Validate API io descriptors are supported for Spark UDF
    # if api.input.__class__ not in SUPPORT_INPUT_IO_DESCRIPTORS:
    #     raise BentoMLException(f"Service API input type {api.input.__class__} is not supported for Spark UDF conversion")
    # if api.output.__class__ not in SUPPORT_OUTPUT_IO_DESCRIPTORS:
    #     raise BentoMLException(f"Service API output type {api.output.__class__} is not supported for Spark UDF conversion")

    # Distribute Bento file to worker nodes
    _distribute_bento(spark, get_bento(bento_tag))

    process = _get_process(bento_tag, api_name)

    if api.output._dtype is None:
        raise BentoMLException(f"Output descriptor for {api_name} must specify a dtype.")

    return pandas_udf(process, returnType=api.output._dtype)
