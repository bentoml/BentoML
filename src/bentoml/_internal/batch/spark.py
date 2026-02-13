from __future__ import annotations

import logging
import os.path
import tempfile
import typing as t
from typing import TYPE_CHECKING

from bentoml._internal.utils import reserve_free_port

from ...bentos import import_bento
from ...bentos import serve
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..bento import Bento
from ..service.loader import load_bento
from ..tag import Tag

try:
    from pyspark.files import SparkFiles
    from pyspark.sql.types import StructType
except ImportError:  # pragma: no cover (trivial error)
    raise MissingDependencyException(
        "'pyspark' is required in order to use module 'bentoml.spark', install pyspark with 'pip install pyspark'. For more information, refer to https://spark.apache.org/docs/latest/api/python/"
    )

if TYPE_CHECKING:
    import pyspark.sql.dataframe
    import pyspark.sql.session

    RecordBatch: t.TypeAlias = t.Any  # pyarrow doesn't have type annotations


logger = logging.getLogger(__name__)


def _is_legacy_service(svc: t.Any) -> bool:
    """Check if the service uses the legacy InferenceAPI interface."""
    from ..service.service import Service

    return isinstance(svc, Service)


def _distribute_bento(spark: pyspark.sql.session.SparkSession, bento: Bento) -> str:
    temp_dir = tempfile.mkdtemp()
    export_path = bento.export(temp_dir)
    spark.sparkContext.addFile(export_path)
    return os.path.basename(export_path)


def _load_bento_spark(bento_tag: Tag):
    """
    load Bento from local bento store or the SparkFiles directory
    """
    try:
        return load_bento(bento_tag)
    except Exception:
        # Use the default Bento export file name. This relies on the implementation
        # of _distribute_bento to use default Bento export file name.
        bento_path = SparkFiles.get(f"{bento_tag.name}-{bento_tag.version}.bento")
        if not os.path.isfile(bento_path):
            raise

        import_bento(bento_path)
        return load_bento(bento_tag)


def _result_to_record_batch(result: t.Any) -> t.Any:
    """Convert an API result to a pyarrow RecordBatch."""
    import numpy as np
    import pandas as pd
    import pyarrow

    if isinstance(result, pyarrow.RecordBatch):
        return result
    elif isinstance(result, pyarrow.Table):
        return result.to_batches()[0]
    elif isinstance(result, pd.DataFrame):
        return pyarrow.RecordBatch.from_pandas(result)
    elif isinstance(result, pd.Series):
        return pyarrow.RecordBatch.from_pandas(result.to_frame())
    elif isinstance(result, np.ndarray):
        return pyarrow.RecordBatch.from_arrays(
            [pyarrow.array(result)], names=["output"]
        )
    elif isinstance(result, (list, tuple)):
        return pyarrow.RecordBatch.from_arrays(
            [pyarrow.array(result)], names=["output"]
        )
    elif isinstance(result, dict):
        arrays = [pyarrow.array(v) if not isinstance(v, pyarrow.Array) else v
                  for v in result.values()]
        return pyarrow.RecordBatch.from_arrays(arrays, names=list(result.keys()))
    else:
        return pyarrow.RecordBatch.from_arrays(
            [pyarrow.array([result])], names=["output"]
        )


def _get_process(
    bento_tag: Tag, api_name: str
) -> t.Callable[[t.Iterable[RecordBatch]], t.Generator[RecordBatch, None, None]]:
    def process(
        iterator: t.Iterable[RecordBatch],
    ) -> t.Generator[RecordBatch, None, None]:
        svc = _load_bento_spark(bento_tag)

        assert api_name in svc.apis, (
            "An error occurred transferring the Bento to the Spark worker."
        )
        api = svc.apis[api_name]

        # start bento server
        with reserve_free_port() as port:
            pass

        server = serve(bento_tag, port=port)

        if _is_legacy_service(svc):
            # Legacy service using InferenceAPI with IO descriptors
            from ...client import Client
            from ...client import HTTPClient

            assert api.func is not None, "Inference API function not defined"
            Client.wait_until_server_ready("localhost", port, 30)
            client = HTTPClient(svc, server.url)

            for batch in iterator:
                func_input = api.input.from_arrow(batch)
                func_output = client.call(api_name, func_input)
                yield api.output.to_arrow(func_output)
        else:
            # New-style service using @bentoml.service() with APIMethod
            from _bentoml_impl.client import SyncHTTPClient

            client = SyncHTTPClient(server.url, server_ready_timeout=30)
            try:
                for batch in iterator:
                    df = batch.to_pandas()
                    input_fields = list(api.input_spec.model_fields.keys())

                    if len(input_fields) == 1:
                        # Single input parameter: pass the whole DataFrame
                        # converted to the appropriate type
                        field_name = input_fields[0]
                        func_output = client.call(
                            api_name, **{field_name: df.to_numpy()}
                        )
                    else:
                        # Multiple input parameters: map DataFrame columns to
                        # function parameter names
                        kwargs = {}
                        for field in input_fields:
                            if field in df.columns:
                                kwargs[field] = df[field].tolist()
                            else:
                                raise BentoMLException(
                                    f"Column '{field}' required by API '{api_name}' "
                                    f"not found in input DataFrame. Available columns: "
                                    f"{list(df.columns)}"
                                )
                        func_output = client.call(api_name, **kwargs)

                    yield _result_to_record_batch(func_output)
            finally:
                client.close()

    return process


def run_in_spark(
    bento: Bento,
    df: pyspark.sql.dataframe.DataFrame,
    spark: pyspark.sql.session.SparkSession,
    api_name: str | None = None,
    output_schema: StructType | None = None,
) -> pyspark.sql.dataframe.DataFrame:
    """
    Run BentoService inference API in Spark.

    The API to run must accept batches as input and return batches as output.

    Args:
        bento:
            The bento containing the inference API to run.
        df:
            The input DataFrame to run the inference API on.
        spark:
            The spark session to use to run the inference API.
        api_name:
            The name of the inference API to run. If not provided, there must be only one API contained
            in the bento; that API will be run.
        output_schema:
            The Spark schema of the output DataFrame. If not provided, BentoML will attempt to infer the
            schema from the output descriptor of the inference API. For new-style services using
            ``@bentoml.service()``, this parameter is required.

    Returns:
        The result of the inference API run on the input ``df``.

    Examples
    --------

    .. code-block:: python

        >>> import bentoml
        >>> import pyspark
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.types import StructType, StructField, StringType

        >>> spark = SparkSession.builder.getOrCreate()
        >>> schema = StructType([
        ...     StructField("name", StringType(), True),
        ...     StructField("age", StringType(), True),
        ... ])
        >>> df = spark.createDataFrame([("John", 30), ("Mike", 25), ("Sally", 40)], schema)

        >>> bento = bentoml.get("my_service:latest")
        >>> results = bentoml.batch.run_in_spark(bento, df, spark)
        >>> results.show()
        +-----+---+
        | name|age|
        +-----+---+
        |John |30 |
        +-----+---+
    """
    svc = load_bento(bento)

    if api_name is None:
        if len(svc.apis) != 1:
            raise BentoMLException(
                f'Bento "{bento.tag}" has multiple APIs ({svc.apis.keys()}), specify which API should be run, e.g.: bentoml.batch.run_in_spark("my_service:latest", df, spark, api_name="predict")'
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

    if output_schema is None:
        if _is_legacy_service(svc):
            output_schema = api.output.spark_schema()
        else:
            raise BentoMLException(
                "output_schema is required when using run_in_spark with new-style "
                "services (using @bentoml.service() decorator). Please provide a "
                "pyspark StructType schema for the output DataFrame."
            )

    return df.mapInArrow(process, output_schema)
