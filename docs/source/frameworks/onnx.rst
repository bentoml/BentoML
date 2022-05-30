====
ONNX
====

Users can now use ONNX with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import math

   import bentoml
   import torch

   import numpy as np
   import torch.nn as nn

   class ExtendedModel(nn.Module):
      def __init__(self, D_in, H, D_out):
            # In the constructor we instantiate two nn.Linear modules and assign them as
            #  member variables.
            super(ExtendedModel, self).__init__()
            self.linear1 = nn.Linear(D_in, H)
            self.linear2 = nn.Linear(H, D_out)

      def forward(self, x, bias):
            # In the forward function we accept a Tensor of input data and an optional bias
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred + bias


   N, D_in, H, D_out = 64, 1000, 100, 1
   x = torch.randn(N, D_in)
   model = ExtendedModel(D_in, H, D_out)

   input_names = ["x", "bias"]
   output_names = ["output1"]

   tmpdir = "/tmp/model"
   model_path = os.path.join(tmpdir, "test_torch.onnx")
   torch.onnx.export(
      model,
      (x, torch.Tensor([1.0])),
      model_path,
      input_names=input_names,
      output_names=output_names,
   )

   # `save` a ONNX model to BentoML modelstore:
   tag = bentoml.onnx.save("onnx_model", model_path, model_store=modelstore)
   bias1, bias2 = bias_pair

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the given model back:
   loaded = bentoml.onnx.load("onnx_model")

   # Run a given model under `Runner` abstraction with `load_runner`
   r1 = bentoml.onnx.load_runner(tag)

   r2 = bentoml.onnx.load_runner(tag)

   res1 = r1.run_batch(x, np.array([bias1]).astype(np.float32))[0][0].item()
   res2 = r2.run_batch(x, np.array([bias2]).astype(np.float32))[0][0].item()

   # tensor to float may introduce larger errors, so we bump rel_tol
   # from 1e-9 to 1e-6 just in case
   assert math.isclose(res1 - res2, bias1 - bias2, rel_tol=1e-6)

.. note::

   You can find more examples for **ONNX** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.onnx

.. autofunction:: bentoml.onnx.save

.. autofunction:: bentoml.onnx.load

.. autofunction:: bentoml.onnx.load_runner
