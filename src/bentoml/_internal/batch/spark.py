from __future__ import annotations

import logging
import os.path
import tempfile
import typing as t
from typing import TYPE_CHECKING

from bentoml._internal.utils import reserve_free_port

from ...bentos import import_bento
from ...bentos import serve
from ...client import Client
from ...client import HTTPClient
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


def _is_legacy_service(svc: t.Any) -> bool:
    from bentoml._internal.service.service import Service as LegacyService

    return isinstance(svc, LegacyService)


def _result_to_record_batch(
    results: list[t.Any], output_schema: StructType
) -> RecordBatch:
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    if not results:
        return pa.RecordBatch.from_pandas(
            pd.DataFrame({n: [] for n in output_schema.names})
        )
    first = results[0]
    if isinstance(first, pa.RecordBatch):
        return first
    if isinstance(first, pa.Table):
        return first.to_batches()[0]
    if isinstance(first, pd.DataFrame):
        return pa.RecordBatch.from_pandas(first)
    if isinstance(first, np.ndarray):
        return pa.RecordBatch.from_pandas(
            pd.DataFrame({output_schema.names[0]: [r.tolist() for r in results]})
        )
    if isinstance(first, list):
        if isinstance(first[0], dict) if first else False:
            return pa.RecordBatch.from_pandas(pd.concat([pd.DataFrame(r) for r in results], ignore_index=True))
        return pa.RecordBatch.from_pandas(
            pd.DataFrame({output_schema.names[0]: list(results)})
        )
    if isinstance(first, dict):
        return pa.RecordBatch.from_pandas(pd.DataFrame(results))
    return pa.RecordBatch.from_pandas(
        pd.DataFrame({output_schema.names[0]: results})
    )


def _get_process(
    bento_tag: Tag,
    api_name: str,
    output_schema: StructType | None = None,
) -> t.Callable[[t.Iterable[RecordBatch]], t.Generator[RecordBatch, None, None]]:
    def process(
        iterator: t.Iterable[RecordBatch],
    ) -> t.Generator[RecordBatch, None, None]:
        svc = _load_bento_spark(bento_tag)

        assert api_name in svc.apis, (
            "An error occurred transferring the Bento to the Spark worker."
        )
        api = svc.apis[api_name]
        assert api.func is not None, "Inference API function not defined"

        # start bento server
        with reserve_free_port() as port:
            pass

        server = serve(bento_tag, port=port)
        Client.wait_until_server_ready("localhost", port, 30)

        if _is_legacy_service(svc):
            client = HTTPClient(svc, server.url)
            for batch in iterator:
                func_input = api.input.from_arrow(batch)
                func_output = client.call(api_name, func_input)
                yield api.output.to_arrow(func_output)
        else:
            import pandas as pd

            from _bentoml_impl.client import SyncHTTPClient as NewSyncHTTPClient

            with NewSyncHTTPClient(server.url, timeout=300) as client:
                for batch in iterator:
                    pdf = batch.to_pandas()
                    row_results = []
                    for _, row in pdf.iterrows():
                        kwargs = row.to_dict()
                        row_results.append(client.call(api_name, **kwargs))
                    yield _result_to_record_batch(row_results, output_schema)

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
            schema from the output descriptor of the inference API.

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

    if _is_legacy_service(svc):
        if output_schema is None:
            output_schema = api.output.spark_schema()
        process = _get_process(bento.tag, api_name)
    else:
        if output_schema is None:
            raise BentoMLException(
                "'output_schema' is required for services defined with the "
                "'@bentoml.service()' decorator, since the output schema "
                "cannot be automatically inferred. Please provide an "
                "'output_schema' argument when calling 'run_in_spark'."
            )
        process = _get_process(bento.tag, api_name, output_schema)

    return df.mapInArrow(process, output_schema)
