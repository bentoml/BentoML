import importlib
import json
import logging
import os

from bentoml.artifact import BentoServiceArtifact
from bentoml.service_env import BentoServiceEnv
from bentoml.exceptions import (
    MissingDependencyException, InvalidArgument, BentoMLException
)

logger = logging.getLogger(__name__)


class PysparkModelArtifact(BentoServiceArtifact):
    # TODO: Write docstring with appropriate usage example
    """

    """

    def __init__(self, name, spark_version=None):
        super(PysparkModelArtifact, self).__init__(name)
        self._model = None
        # TODO: Find a use for this attribute (e.g. Does the spark version
        #  affect how saving/loading is conducted?)
        self._spark_version = spark_version

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def set_dependencies(self, env: BentoServiceEnv):
        # TODO: Write more specific warning when packaging details are
        #  hashed out. The warning below just conveys the idea.
        logger.warning(
            "BentoML by default does not include Spark JAR (Java archive) "
            "dependencies. PySpark requires these dependencies to load models "
            "and perform inference, so be sure to build them locally before "
            "importing your model."
        )
        env.add_pip_dependencies_if_missing(['pyspark'])

    def pack(self, model):
        try:
            import pyspark
        except ImportError:
            raise MissingDependencyException(
                "pyspark package is required to use PysparkModelArtifact"
            )

        # TODO: Confirm supported model typing. Possible nuances:
        #  - spark.ml vs spark.mllib
        #  - Model vs. Estimator
        if not isinstance(model, pyspark.ml.Model):
            raise InvalidArgument(
                "PysparkModelArtifact can only pack type 'pyspark.ml.Model'"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            import pyspark
        except ImportError:
            raise MissingDependencyException(
                "pyspark package is required to use PysparkModelArtifact"
            )

        model_path = self._file_path(path)
        metadata_path = model_path+"/metadata/part-00000"

        # TODO: Verify that "part-00000" is a reliable metadata filename
        with open(metadata_path) as f:
            metadata = json.load(f)

        if "class" not in metadata:
            raise BentoMLException("Malformed metadata file.")

        # TODO: Verify cases where model type can/can't be inferred this way
        # Follows convention ["org", "apache", "spark", "ml",  [...], "Model"]
        model_class_str_list = metadata["class"].split(".")
        module_name = ".".join(["pyspark"] + model_class_str_list[3:-1])
        model_type = model_class_str_list[-1]

        # Use load method specific to the pyspark.ml model class
        ModelClass = getattr(importlib.import_module(module_name), model_type)
        logger.info(f"Loading {ModelClass} model from {model_path}.")
        model = ModelClass.load(model_path)

        # TODO: Confirm supported model typing. Possible nuances:
        #  - spark.ml vs spark.mllib (pyspark.mllib.Model?)
        if not isinstance(model, pyspark.ml.Model):
            raise InvalidArgument(
                f"Expecting PysparkModelArtifact loaded object type to be "
                f"'pyspark.ml.Model' but actually it is {type(model)}."
            )

        return self.pack(model)

    def get(self):
        return self._model

    def save(self, dst):
        if not self._model:
            # TODO: Find appropriate Exception type
            raise Exception(
                "Model file not packed before attempting to save."
            )

        return self._model.save(self._file_path(dst))
