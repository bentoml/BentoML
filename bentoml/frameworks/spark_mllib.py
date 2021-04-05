import importlib
import json
import logging
import os

from bentoml.exceptions import BentoMLException, MissingDependencyException
from bentoml.service import BentoServiceArtifact, BentoServiceEnv


class PySparkModelArtifact(BentoServiceArtifact):
    """
    PySparkModelArtifact allows you to use the spark.ml and spark.mllib APIs with Bento

    Example usage:

    >>> from pyspark.sql import SparkSession
    >>> 
    >>> spark = SparkSession.builder.getOrCreate()
    >>>
    >>> # Load training data
    >>> training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    >>> lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    >>> # Fit the model
    >>> lrModel = lr.fit(training)
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.sklearn import PySparkModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([PySparkModelArtifact('model')])
    >>> class SparkModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         result = self.artifacts.model.predict(df)
    >>>         return result
    >>>
    >>> svc = SparkModelService()
    >>>
    >>> # Pack directly with sklearn model object
    >>> svc.pack('model', model_to_save)
    """

    def __init__(self, name: str, spark=None):
        super().__init__(name)
        self._model = None
        self._sc = None
        self.spark = spark

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_package("pyspark")
        env.add_conda_dependencies(["openjdk"])

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

        spark = self.spark or SparkSession.builder.appName('BentoService').getOrCreate()

        model_data_path = os.path.join(path, self.name)
        metadata_path = os.path.join(path, self.name, "metadata/part-00000")

        try:
            with open(metadata_path, "r") as metadata:
                metadata = json.load(metadata)
        except IOError:
            raise BentoMLException(
                "Incorrectly serialized model was loaded. Unable to load metadata"
            )
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
