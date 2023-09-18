from __future__ import annotations

import typing as t
import logging
import os.path
import tempfile
from typing import TYPE_CHECKING

from bentoml._internal.utils import reserve_free_port

from ..tag import Tag
from ..bento import Bento
from ...bentos import serve
from ...bentos import import_bento
from ...client import Client
from ...client import HTTPClient
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..service.loader import load_bento

try:
    from pyspark.files import SparkFiles
    from pyspark.sql.types import StructType
except ImportError:  # pragma: no cover (trivial error)
    raise MissingDependencyException(
        "'pyspark' is required in order to use module 'bentoml.spark', install pyspark with 'pip install pyspark'. For more information, refer to https://spark.apache.org/docs/latest/api/python/"
    )

if TYPE_CHECKING:
    import pyspark.sql.session
    import pyspark.sql.dataframe

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


def _get_process(
    bento_tag: Tag, api_name: str
) -> t.Callable[[t.Iterable[RecordBatch]], t.Generator[RecordBatch, None, None]]:
    def process(
        iterator: t.Iterable[RecordBatch],
    ) -> t.Generator[RecordBatch, None, None]:
        svc = _load_bento_spark(bento_tag)

        assert (
            api_name in svc.apis
        ), "An error occurred transferring the Bento to the Spark worker."
        inference_api = svc.apis[api_name]
        assert inference_api.func is not None, "Inference API function not defined"

        # start bento server
        with reserve_free_port() as port:
            pass

        server = serve(bento_tag, port=port)
        Client.wait_until_server_ready("localhost", server.port, 30)
        client = HTTPClient(svc, f"http://localhost:{server.port}")

        for batch in iterator:
            func_input = inference_api.input.from_arrow(batch)
            func_output = client.call(api_name, func_input)
            yield inference_api.output.to_arrow(func_output)

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

    process = _get_process(bento.tag, api_name)

    if output_schema is None:
        output_schema = api.output.spark_schema()

    return df.mapInArrow(process, output_schema)
