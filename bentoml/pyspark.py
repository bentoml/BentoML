import importlib
import json
import logging
import os
import typing as t

import bentoml._internal.constants as _const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import GenericDictType, PathType
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException

logger = logging.getLogger(__name__)

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="pyspark.mllib",
    module=__name__,
    inst="First install Apache Spark, https://spark.apache.org/downloads.html."
    " Then run `pip install pyspark`",
)

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import pyspark
    import pyspark.ml as ml
    import pyspark.sql as sql
else:
    pyspark = LazyLoader("pyspark", globals(), "pyspark", exc_msg=_exc)
    ml = LazyLoader("ml", globals(), "pyspark.ml", exc_msg=_exc)
    sql = LazyLoader("sql", globals(), "pyspark.sql", exc_msg=_exc)

# NOTE: the usage of SPARK_SESSION_NAMESPACE is to provide a consistent session
#  among imports if users need to use SparkSession.
SPARK_SESSION_NAMESPACE: str = "PySparkMLlibModel"

DEPRECATION_MLLIB_WARNING: str = """\
{model} is using the older library `pyspark.mllib`.
Consider to upgrade your model to use `pyspark.ml`.
BentoML will still try to load {model} with `pyspark.sql.SparkSession`,
but expect unintended behaviour.
"""


class PySparkMLlibModel(Model):
    """
    Model class for saving/loading :obj:`pyspark` models
    using :obj:`pyspark.ml` and :obj:`pyspark.mllib`

    Args:
        model (`pyspark.ml.Model`):
            Every PySpark model is of type :obj:`pyspark.ml.Model`
        spark_session (`pyspark.sql.SparkSession`, `optional`, default to `None`):
            Optional SparkSession used to load PySpark model representation.
        metadata (`GenericDictType`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`pyspark` is required by PySparkMLlibModel

    .. WARNING::

        :obj:`spark_session` should only be used when your current model is running
        older version of `pyspark.ml` (`pyspark.mllib`). Consider to upgrade your mode
        beforehand to use `pyspark.ml`.

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    _model: "pyspark.ml.Model"

    def __init__(
        self,
        model: "pyspark.ml.Model",
        spark_session: t.Optional["pyspark.sql.SparkSession"] = None,
        metadata: t.Optional[GenericDictType] = None,
    ):
        super(PySparkMLlibModel, self).__init__(model, metadata=metadata)
        # NOTES: referred to docstring, spark_session is mainly used
        #  for backward compatibility.
        self._spark_sess = spark_session

    @classmethod
    def load(cls, path: PathType) -> "pyspark.ml.Model":

        model_path: str = str(os.path.join(path, MODEL_NAMESPACE))

        # NOTE (future ref): A large model metadata might
        #  comprise of multiple `part` files, instead of assigning,
        #  loop through the directory.
        metadata_path: str = str(os.path.join(model_path, "metadata/part-00000"))

        try:
            with open(metadata_path, "r") as meta_file:
                metadata = json.load(meta_file)
        except IOError:
            raise BentoMLException(
                "Incorrectly serialized model was loaded. Unable to load metadata"
            )
        if "class" not in metadata:
            raise BentoMLException("malformed metadata file.")
        model_class = metadata["class"]

        # process imports from metadata
        stripped_apache_module: t.List[str] = model_class.split(".")[2:]
        py_module = "py" + ".".join(stripped_apache_module[:-1])  # skip org.apache
        class_name = stripped_apache_module[-1]

        loaded_model = getattr(importlib.import_module(py_module), class_name)
        if not issubclass(loaded_model, ml.Model):
            logger.warning(DEPRECATION_MLLIB_WARNING.format(model=loaded_model))
            _spark_sess = sql.SparkSession.builder.appName(
                SPARK_SESSION_NAMESPACE
            ).getOrCreate()
            model = loaded_model.load(_spark_sess.sparkContext, model_path)
        else:
            model = loaded_model.load(model_path)

        return model

    def save(self, path: PathType) -> None:
        if not isinstance(self._model, ml.Model):
            logger.warning(DEPRECATION_MLLIB_WARNING.format(model=self._model))
            self._model.save(self._spark_sess, os.path.join(path, MODEL_NAMESPACE))
        else:
            self._model.save(os.path.join(path, MODEL_NAMESPACE))
