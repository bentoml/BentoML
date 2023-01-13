from __future__ import annotations

import typing as t
import logging
import os.path
import tempfile
from typing import TYPE_CHECKING

from ..tag import Tag
from ..bento import Bento
from ...bentos import serve
from ...bentos import import_bento
from ...client import Client
from ...client import HTTPClient
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..service.loader import load_bento
from ..service.loader import _load_bento as load_bento_service  # type: ignore

try:
    from pyspark.files import SparkFiles
    from pyspark.sql.types import StructType
    from pyspark.sql.dataframe import DataFrame
except ImportError:  # pragma: no cover (trivial error)
    raise MissingDependencyException(
        '"pyspark" is required in order to use module `bentoml.spark`, install pyspark with `pip install pyspark`. For more information, refer to https://spark.apache.org/docs/latest/api/python/'
    )

try:
    import pyarrow
except ImportError:  # pragma: no cover (trivial error)
    raise MissingDependencyException(
        '"pyspark" is required in order to use module `bentoml.spark`, install pyspark with `pip install pyspark`. For more information, refer to https://spark.apache.org/docs/latest/api/python/'
    )

if TYPE_CHECKING:
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


def _get_process(
    bento_tag: Tag, api_name: str
) -> t.Callable[
    [t.Iterator[pyarrow.RecordBatch]], t.Generator[pyarrow.RecordBatch, None, None]
]:
    def process(
        iterator: t.Iterator[pyarrow.RecordBatch],
    ) -> t.Generator[pyarrow.RecordBatch, None, None]:

        svc = _load_bento(bento_tag)

        assert (
            api_name in svc.apis
        ), "An error occurred transferring the Bento to the Spark worker; see <something>."
        inference_api = svc.apis[api_name]
        assert inference_api.func is not None, "Inference API function not defined"

        # start bento server
        server = serve(bento_tag)
        Client.wait_until_server_is_ready("localhost", server.port, 30)
        client = HTTPClient(svc, f"http://localhost:{server.port}")

        for batch in iterator:
            # default batch size = 10,000
            func_input = inference_api.input.from_arrow(batch)
            func_output = client.call(api_name, func_input)
            yield inference_api.output.to_arrow(func_output)

    return process  # type: ignore  # process type evaluated at runtime


def run_in_spark(
    bento: Bento,
    df: DataFrame,
    spark: SparkSession,
    api_name: str | None = None,
    output_schema: StructType | None = None,
) -> DataFrame:
    svc = load_bento_service(bento, False)

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
