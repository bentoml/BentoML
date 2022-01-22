onnx-mlir
---------
| The Open Neural Network Exchange implementation in MLIR - `Source <https://github.com/onnx/onnx-mlir>`_

Users can now use onnx-mlir with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import sys
   import os
   import subprocess

   import bentoml
   import tensorflow as tf

   sys.path.append("/workdir/onnx-mlir/build/Debug/lib/")

   from PyRuntime import ExecutionSession

   class NativeModel(tf.Module):
      def __init__(self):
            super().__init__()
            self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
            self.dense = lambda inputs: tf.matmul(inputs, self.weights)

      @tf.function(
            input_signature=[tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="inputs")]
      )
      def __call__(self, inputs):
            return self.dense(inputs)

   directory = "/tmp/model"
   model = NativeModel()
   tf.saved_model.save(model, directory)

   model_path = os.path.join(directory, "model.onnx")
   command = [
      "python",
      "-m",
      "tf2onnx.convert",
      "--saved-model",
      directory,
      "--output",
      model_path,
   ]
   docker_proc = subprocess.Popen(  # noqa
      command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir, text=True
   )
   stdout, stderr = docker_proc.communicate()

   sys.path.append("/workdir/onnx-mlir/build/Debug/lib/")
   model_location = os.path.join(directory, "model.onnx")
   command = ["./onnx-mlir", "--EmitLib", model_location]
   onnx_mlir_loc = "/workdir/onnx-mlir/build/Debug/bin"

   docker_proc = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      cwd=onnx_mlir_loc,
   )
   stdout, stderr = docker_proc.communicate()

   model_path = os.path.join(directory, "model.so")

   # `save` a ONNX model to BentoML modelstore:
   tag = bentoml.onnxmlir.save("compiled_model", model)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the given model back:
   loaded = bentoml.onnxmlir.load("compiled_model")

   # Run a given model under `Runner` abstraction with `load_runner`
   runner = bentoml.onnxmlir.load_runner("compiled_model:latest")
   res = runner.run_batch(np.array([[1,2,3]]).astype(np.float64))

.. note::
   You can find more examples for **ONNX** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.onnxmlir

.. autofunction:: bentoml.onnxmlir.save

.. autofunction:: bentoml.onnxmlir.load

.. autofunction:: bentoml.onnxmlir.load_runner