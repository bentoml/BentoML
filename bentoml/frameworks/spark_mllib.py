import importlib
import json
import logging
import os

from bentoml.exceptions import BentoMLException, MissingDependencyException
from bentoml.service import BentoServiceArtifact, BentoServiceEnv


class PySparkModelArtifact(BentoServiceArtifact):
    def __init__(self, name: str):
        super().__init__(name)
        self._model = None
        self._sc = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_package("pyspark")

    def pack(
        self, model, metadata: dict = None, sc=None
    ):  # pylint:disable=arguments-differ
        self._sc = sc
        self._model = model

        return self

    def save(self, dst):
        model = self._model
        save_path = os.path.join(dst, self.name)

        try:
            from pyspark.ml.base import Model
        except ImportError:
            raise MissingDependencyException(
                "pyspark is required to use the PySparkModelArtifact"
            )

        if isinstance(model, Model):
            model: Model = model
            model.save(save_path)
        else:
            model.save(self._sc, save_path)

    def load(self, path):
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            raise MissingDependencyException(
                "pyspark is required to use the PySparkModelArtifact"
            )

        spark = SparkSession.builder.appName('BentoService').getOrCreate()

        model_data_path = os.path.join(path, self.name)
        metadata_path = os.path.join(path, self.name, "metadata/part-00000")

        with open(metadata_path, "r") as metadata:
            metadata = json.load(metadata)
        if "class" not in metadata:
            raise BentoMLException("Malformed metadata file.")
        model_class = metadata["class"]

        logger = logging.getLogger(self.name)
        logger.info("Loading %s model from %s" % (model_class, model_data_path))
        splits = model_class.split(".")[2:]  # skip the org.apache
        module = "py" + ".".join(splits[:-1])
        class_name = splits[-1]
        # noinspection PyPep8Naming
        ModelClass = getattr(importlib.import_module(module), class_name)
        try:
            from pyspark.ml import Model
        except ImportError:
            raise MissingDependencyException(
                "pyspark is required to use the PySparkModelArtifact"
            )

        if issubclass(ModelClass, Model):
            model = ModelClass.load(model_data_path)
        else:
            model = ModelClass.load(spark.sparkContext, model_data_path)

        return self.pack(model)

    def get(self):
        return self._model
