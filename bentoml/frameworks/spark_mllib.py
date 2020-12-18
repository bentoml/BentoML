import importlib
import json
import logging
import os
import typing
from typing import Optional

from bentoml.exceptions import MissingDependencyException, BentoMLException
from bentoml.service import BentoServiceArtifact

if typing.TYPE_CHECKING:
    import pyspark
    from pyspark.ml.base import Model


class PySparkModelArtifact(BentoServiceArtifact):

    def __init__(self, name: str):
        super().__init__(name)
        self._model = None
        self._sc: Optional[pyspark.SparkContext] = None

    def pack(self, model, metadata: dict = None, sc: pyspark.SparkContext = None):  # pylint:disable=arguments-differ
        try:
            import pyspark
            from pyspark.ml.base import Model
        except ImportError:
            raise MissingDependencyException(
                "pyspark is required to use PySparkModelArtifact"
            )
        self._sc = sc
        self._model = model

        return self

    def save(self, dst):
        model = self._model
        save_path = os.path.join(dst, self.name)

        if isinstance(model, pyspark.ml.base.Model):
            model: pyspark.ml.base.Model = model
            model.save(save_path)
        else:
            model.save(self._sc, save_path)

    def load(self, path):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName('BentoService').getOrCreate()

        model_data_path = os.path.join(path, "pyspark_model_data")
        metadata_path = os.path.join(path, "metadata.json")

        with open(metadata_path, "r") as metadata:
            metadata = json.load(metadata)
        if "model_class" not in metadata:
            raise BentoMLException("Malformed metadata file.")
        model_class = metadata["model_class"]

        logger = logging.getLogger(self.name)
        logger.info("Loading %s model from %s" % (model_class, model_data_path))
        splits = model_class.split(".")
        module = ".".join(splits[:-1])
        class_name = splits[-1]
        ModelClass = getattr(importlib.import_module(module), class_name)
        if issubclass(ModelClass,
                      pyspark.ml.pipeline.PipelineModel) or issubclass(
            ModelClass, pyspark.ml.base.Model):
            model = ModelClass.load(model_data_path)
        else:
            model = ModelClass.load(spark.sparkContext, model_data_path)

        return self.pack(model)
