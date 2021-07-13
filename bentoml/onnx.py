import logging
import os
import pathlib
import shutil

from bentoml.exceptions import (
    BentoMLException,
    InvalidArgument,
    MissingDependencyException,
)
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

logger = logging.getLogger(__name__)


SUPPORTED_ONNX_BACKEND = ["onnxruntime", "onnxruntime-gpu"]


def _is_path_like(path):
    return isinstance(path, (str, bytes, pathlib.Path, os.PathLike))


def _is_onnx_model_file(path):
    return (
        _is_path_like(path)
        and os.path.isfile(path)
        and str(path).lower().endswith(".onnx")
    )


class OnnxModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving and loading ONNX Model

    Args:
        name (string): Name of the artifact
        backend (string): Name of ONNX inference runtime. ["onnxruntime",
                            "onnxruntime-gpu"]

    Raises:
        MissingDependencyException: onnx package is required for packing a ModelProto
                                    object
        NotImplementedError: {backend} as onnx runtime is not supported at the moment

    Example usage:

    >>> # Train a model.
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> clr = RandomForestClassifier()
    >>> clr.fit(X_train, y_train)
    >>> # Convert into ONNX format
    >>> from skl2onnx import convert_sklearn
    >>> from skl2onnx.common.data_types import FloatTensorType
    >>> initial_type = [('float_input', FloatTensorType([None, 4]))]
    >>>
    >>> onnx_model = convert_sklearn(clr, initial_types=initial_type)
    >>> with open("rf_iris.onnx", "wb") as f:
    >>>     f.write(onnx_model.SerializeToString())
    >>>
    >>>
    >>> import numpy
    >>> import bentoml
    >>> from bentoml.frameworks.onnx import OnnxModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([OnnxModelArtifact('model', backend='onnxruntime')])
    >>> class OnnxIrisClassifierService(bentoml.BentoService):
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         input_data = df.to_numpy().astype(numpy.float32
    >>>         input_name = self.artifacts.model.get_inputs()[0].name
    >>>         output_name = self.artifacts.model.get_outputs()[0].name
    >>>         return self.artifacts.model.run(
    >>>                     [output_name], {input_name: input_data}
    >>>                )[0]
    >>>
    >>> svc = OnnxIrisClassifierService()
    >>>
    >>> # Option one: pack with path to model on local system
    >>> svc.pack('model', './rf_iris.onnx')
    >>>
    >>> # Option two: pack with ONNX model object
    >>> # svc.pack('model', onnx_model)
    >>>
    >>> # Save BentoService
    >>> svc.save()
    """

    def __init__(self, name, backend="onnxruntime"):
        super().__init__(name)
        if backend not in SUPPORTED_ONNX_BACKEND:
            raise BentoMLException(
                f'"{backend}" runtime is currently not supported for OnnxModelArtifact'
            )
        self.backend = backend
        self._inference_session = None
        self._onnx_model_path = None
        self._model_proto = None

    def _saved_model_file_path(self, base_path):
        return os.path.join(base_path, self.name + ".onnx")

    def pack(
        self, path_or_model_proto, metadata=None
    ):  # pylint:disable=arguments-renamed
        if _is_onnx_model_file(path_or_model_proto):
            self._onnx_model_path = path_or_model_proto
        else:
            try:
                import onnx

                if isinstance(path_or_model_proto, onnx.ModelProto):
                    self._model_proto = path_or_model_proto
                else:
                    raise InvalidArgument(
                        "onnx.ModelProto or a .onnx model file path is required to "
                        "pack an OnnxModelArtifact"
                    )
            except ImportError:
                raise InvalidArgument(
                    "onnx.ModelProto or a .onnx model file path is required to pack "
                    "an OnnxModelArtifact"
                )

        assert self._onnx_model_path or self._model_proto, (
            "Either self._onnx_model_path or self._model_proto has to be initialized "
            "after initializing _OnnxModelArtifactWrapper"
        )

        return self

    def load(self, path):
        return self.pack(self._saved_model_file_path(path))

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            if self.backend == "onnxruntime":
                env.add_pip_packages(["onnxruntime"])
            elif self.backend == "onnxruntime-gpu":
                env.add_pip_packages(["onnxruntime-gpu"])

    def _get_onnx_inference_session(self):
        if self.backend == "onnxruntime" or self.backend == "onnxruntime-gpu":
            try:
                import onnxruntime
            except ImportError:
                raise MissingDependencyException(
                    f'"{self.backend}" package is required for inference with '
                    f'"{self.backend}" as backend"'
                )

            if self._model_proto:
                logger.info(
                    "Initializing onnxruntime InferenceSession with onnx.ModelProto "
                    "instance"
                )
                return onnxruntime.InferenceSession(
                    self._model_proto.SerializeToString()
                )
            elif self._onnx_model_path:
                logger.info(
                    "Initializing onnxruntime InferenceSession from onnx file:"
                    f"'{self._onnx_model_path}'"
                )
                return onnxruntime.InferenceSession(self._onnx_model_path)
            else:
                raise BentoMLException("OnnxModelArtifact in bad state")
        else:
            raise BentoMLException(
                f'"{self.backend}" runtime is currently not supported for '
                f"OnnxModelArtifact"
            )

    def get(self):
        if not self._inference_session:
            self._inference_session = self._get_onnx_inference_session()
        return self._inference_session

    def save(self, dst):
        if self._onnx_model_path:
            shutil.copyfile(self._onnx_model_path, self._saved_model_file_path(dst))
        elif self._model_proto:
            try:
                import onnx
            except ImportError:
                raise MissingDependencyException(
                    '"onnx" package is required for packing with OnnxModelArtifact'
                )
            onnx.save_model(self._model_proto, self._saved_model_file_path(dst))
        else:
            raise InvalidArgument(
                "onnx.ModelProto or a model file path is required to pack an "
                "OnnxModelArtifact"
            )
