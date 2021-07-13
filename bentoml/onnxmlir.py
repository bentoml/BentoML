import os
import shutil

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class OnnxMlirModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving/loading onnx-mlir compiled model and operationalized
    using pyruntime wrapper

    onnx-mlir is a compiler technology that can take an onnx model and lower it
    (using llvm) to an inference library that is optimized and has little external
    dependencies.

    The PyRuntime interface is created during the build of onnx-mlir using pybind.
    See the onnx-mlir supporting documentation for detail.

    Args:
        name (string): Name of the artifact
    Raises:
        MissingDependencyException: PyRuntime must be accessible in path.

    Example usage:

    >>> import bentoml
    >>> from bentoml import env, artifacts, api, BentoService
    >>> from bentoml.adapters import ImageInput
    >>> from bentoml.frameworks.onnxmlir import OnnxMlirModelArtifact
    >>> from bentoml.types import JsonSerializable, InferenceTask, InferenceError
    >>> from bentoml.service.artifacts.common import PickleArtifact
    >>> from typing import List
    >>> import numpy as np
    >>> import pandas as pd
    >>> import sys
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([OnnxMlirModelArtifact('model'), PickleArtifact('labels')])
    >>> class ResNetPredict(BentoService):
    >>>
    >>>     def preprocess(self, input_data):
    >>>         # convert the input data into the float32 input
    >>>         img_data = np.stack(input_data).transpose(0, 3, 1, 2)
    >>>
    >>>         #normalize
    >>>         mean_vec = np.array([0.485, 0.456, 0.406])
    >>>         stddev_vec = np.array([0.229, 0.224, 0.225])
    >>>
    >>>
    >>>         norm_img_data = np.zeros(img_data.shape).astype('float32')
    >>>
    >>>
    >>>         for i in range(img_data.shape[0]):
    >>>             for j in range(img_data.shape[1]):
    >>>                 norm_img_data[i,j,:,:] =
    >>>                 (img_data[i,j,:,:]/255 - mean_vec[j]) / stddev_vec[j]
    >>>
    >>>         #add batch channel
    >>>         norm_img_data = norm_img_data.reshape(-1, 3, 224, 224).astype('float32')
    >>>         return norm_img_data
    >>>
    >>>
    >>>     def softmax(self, x):
    >>>         x = x.reshape(-1)
    >>>         e_x = np.exp(x - np.max(x))
    >>>         return e_x / e_x.sum(axis=0)
    >>>
    >>>     def post_process(self, raw_result):
    >>>         return self.softmax(np.array(raw_result)).tolist()
    >>>
    >>>     @bentoml.api(input=ImageInput(), batch=True)
    >>>     def predict(self, image_ndarrays: List[np.ndarray]) -> List[str]:
    >>>         input_datas = self.preprocess(image_ndarrays)
    >>>
    >>>         outputs = []
    >>>         for i in range(input_datas.shape[0]):
    >>>             raw_result = self.artifacts.model.run(input_datas[i:i+1])
    >>>             result = self.post_process(raw_result)
    >>>             idx = np.argmax(result)
    >>>             sort_idx = np.flip(np.squeeze(np.argsort(result)))
    >>>
    >>>             # return top 5 labels
    >>>             outputs.append(self.artifacts.labels[sort_idx[:5]])
    >>>             return outputs
    >>>
    """

    def __init__(self, name):
        super().__init__(name)
        self._inference_session = None
        self._model_so_path = None

    def _saved_model_file_path(self, base_path):
        self._model_so_path = os.path.join(base_path, self.name + '.so')
        return os.path.join(base_path, self.name + '.so')

    def pack(self, onnxmlir_model_so, metadata=None):
        # pylint:disable=arguments-renamed
        self._model_so_path = onnxmlir_model_so
        return self

    def load(self, path):
        print(path)
        return self.pack(self._saved_model_file_path(path))

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(["numpy"])

    def _get_onnxmlir_inference_session(self):
        try:
            # this has to be able to find the arch and OS specific PyRuntime .so file
            from PyRuntime import ExecutionSession
        except ImportError:
            raise MissingDependencyException(
                "PyRuntime package library must be in python path"
            )
        return ExecutionSession(self._model_so_path, "run_main_graph")

    def get(self):
        if not self._inference_session:
            self._inference_session = self._get_onnxmlir_inference_session()
        return self._inference_session

    def save(self, dst):
        # copies the model .so and places in the version controlled deployment path
        shutil.copyfile(self._model_so_path, self._saved_model_file_path(dst))
