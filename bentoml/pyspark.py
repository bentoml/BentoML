# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import importlib
import json
import logging
import os
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import BentoMLException, MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import pyspark
    import pyspark.ml
    from pyspark.sql import SparkSession
except ImportError:
    raise MissingDependencyException("pyspark is required by PySparkMLlibModel")

logger = logging.getLogger(__name__)

# NOTE: the usage of SPARK_SESSION_NAMESPACE is to provide a consistent session
#  among imports if users need to use SparkSession.
SPARK_SESSION_NAMESPACE: str = "PySparkMLlibModel"

DEPRECATION_MLLIB_WARNING: str = """\
    {model} is using the older library `pyspark.mllib`.
    Consider to upgrade your model to use `pyspark.ml`.
    BentoML will still try to load {model} with `pyspark.sql.SparkSession`,
    but expect unintended behaviour.
    """


class PySparkMLlibModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`pyspark` models
    using :obj:`pyspark.ml` and :obj:`pyspark.mllib`

    Args:
        model (`pyspark.ml.Model`):
            Every PySpark model is of type :obj:`pyspark.ml.Model`
        spark_session (`pyspark.sql.SparkSession`, `optional`, default to `None`):
            Optional SparkSession used to load PySpark model representation.
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
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

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    _model: pyspark.ml.Model

    def __init__(
        self,
        model: pyspark.ml.Model,
        spark_session: t.Optional["SparkSession"] = None,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PySparkMLlibModel, self).__init__(model, metadata=metadata)
        # NOTES: referred to docstring, spark_session is mainly used for backward compatibility
        self._spark_sess = spark_session

    @classmethod
    def load(cls, path: PathType) -> pyspark.ml.Model:
        model_path: str = str(cls.get_path(path))

        # NOTE (future ref): A large model metadata might
        #  comprise of multiple `part` files, instead of assigning,
        #  loop through the directory.
        metadata_path: str = str(os.path.join(model_path, 'metadata/part-00000'))

        try:
            with open(metadata_path, 'r') as meta_file:
                metadata = json.load(meta_file)
        except IOError:
            raise BentoMLException(
                "Incorrectly serialized model was loaded. Unable to load metadata"
            )
        if 'class' not in metadata:
            raise BentoMLException('malformed metadata file.')
        model_class = metadata['class']

        # process imports from metadata
        stripped_apache_module: t.List[str] = model_class.split('.')[2:]
        py_module = 'py' + '.'.join(stripped_apache_module[:-1])  # skip org.apache
        class_name = stripped_apache_module[-1]

        loaded_model = getattr(importlib.import_module(py_module), class_name)
        if not issubclass(loaded_model, pyspark.ml.Model):
            logger.warning(DEPRECATION_MLLIB_WARNING.format(model=loaded_model))
            # fmt: off # noqa: E501
            _spark_sess = SparkSession.builder.appName(
                SPARK_SESSION_NAMESPACE
            ).getOrCreate()
            # fmt: on
            model = loaded_model.load(_spark_sess.sparkContext, model_path)
        else:
            model = loaded_model.load(model_path)

        return model

    def save(self, path: PathType) -> None:
        if not isinstance(self._model, pyspark.ml.Model):
            logger.warning(DEPRECATION_MLLIB_WARNING.format(model=self._model))
            self._model.save(self._spark_sess, self.get_path(path))
        else:
            self._model.save(self.get_path(path))
