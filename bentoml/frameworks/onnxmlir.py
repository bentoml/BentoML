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

class OnnxMlirModelArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading onnx-mlir compiled model and operationalized using pyruntime wrapper

    onnx-mlir is a compiler technology that can take an onnx model and lower it (using llvm) to an inference
    library that is optimized and has little external dependencies.

    The PyRuntime interface is created during the build of onnx-mlir using pybind. 
    See the onnx-mlir supporting documentation for detail. 

    Args:
        name (string): Name of the artifact
    Raises:
        MissingDependencyException: PyRuntime must be accessible in path.

    Example usage:
        import bentoml
        from bentoml import env, artifacts, api, BentoService
        from bentoml.adapters import DataframeInput
        from bentoml.adapters import JsonInput
        from bentoml.frameworks.onnxmlir import OnnxMlirModelArtifact
        import numpy as np
        import pandas as pd
        import sys

        sys.path.insert(0, '/Users/andrewsius.ibm.com/Documents/models/mnist/')
        print(sys.path)

        @bentoml.env(infer_pip_packages=True)
        @bentoml.artifacts([OnnxMlirModelArtifact('model')])
        class MyPredictionService(BentoService):

         A simple prediction service exposing a Onnx-Mlir model


        @bentoml.api(input=DataframeInput(orient='values'), batch=True)
        def predict(self, df: pd.DataFrame):

            An inference API named `predict` with Dataframe input adapter, which defines
            how HTTP requests or CSV files get converted to a pandas Dataframe object as the
            inference API function input

            input_data = df.to_numpy().astype(np.float32)
            print(input_data.shape)
            result = self.artifacts.model.run(input_data)
            return result


        from bcOnnxM import MyPredictionService

            import sys
            sys.path.insert(0, '/Users/andrewsius.ibm.com/Documents/models/mnist/')
            svc = MyPredictionService()
            svc.pack('model', '/Users/andrewsius.ibm.com/Documents/models/mnist/model.so')
            location = svc.save()
            print(location)

    """

    def __init__(self, name):
        super(OnnxMlirModelArtifact, self).__init__(name)
        self._inference_session = None
        self._model_so_path = None

    def _saved_model_file_path(self, base_path):
        self._model_so_path = os.path.join(base_path, self.name + '.so')
        print(self._model_so_path)
        return os.path.join(base_path, self.name + '.so')

    def pack(self, onnxmlir_model_so, metadata=None):  
        print(onnxmlir_model_so)
        self._model_so_path = onnxmlir_model_so
        return self

    def load(self, path):
        print(path)
        return self.pack(self._saved_model_file_path(path))

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(["numpy"])

    def _get_onnxmlir_inference_session(self):
        try:
            from PyRuntime import ExecutionSession  
        except ImportError:
            raise MissingDependencyException(
                "PyRuntime package must be in python path"
            )
        print(self._model_so_path)
        return ExecutionSession(self._model_so_path, "run_main_graph")

    def get(self):
        if not self._inference_session:
            self._inference_session = self._get_onnxmlir_inference_session()
        return self._inference_session

    def save(self, dst):
        shutil.copyfile(self._model_so_path, self._saved_model_file_path(dst))


