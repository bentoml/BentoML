====
ONNX
====

ONNX is an open format built to represent machine learning models. ONNX provides `high interoperability <https://onnx.ai/supported-tools.html#buildModel>`_  among various frameworks, as well as enable machine learning practitioners to maximize models' performance across `different hardware <https://onnx.ai/supported-tools.html#deployModel>`_.

Due to its high interoperability among frameworks, we recommend you to check out the framework integration with ONNX as it will contain specific recommendation and requirements for that given framework.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      - `Quick tutorial about exporting a model from PyTorch to ONNX <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>`_ from official PyTorch documentation.
      - `torch.onnx <https://pytorch.org/docs/stable/onnx.html>`_ from official PyTorch documentation. Pay special attention to section **Avoiding Pitfalls**, **Limitations** and **Frequently Asked Questions**.

   .. tab-item:: TensorFlow
      :sync: tensorflow

      - `tensorflow-onnx (tf2onnx) <https://github.com/onnx/tensorflow-onnx>`_ documentation.

   .. tab-item:: Scikit Learn
      :sync: sklearn

      TODO


Compatibility
-------------

BentoML currently only support `ONNX Runtime
<https://onnxruntime.ai>`_ as ONNX backend. BentoML requires either
``onnxruntime>=1.9`` or ``onnxruntime-gpu>=1.9`` to be installed.


Converting model frameworks to ONNX format
------------------------------------------

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      First, let’s create a SuperResolution model in PyTorch.

      .. code-block:: python

	 import torch.nn as nn
	 import torch.nn.init as init

	 class SuperResolutionNet(nn.Module):
	     def __init__(self, upscale_factor, inplace=False):
		 super(SuperResolutionNet, self).__init__()

		 self.relu = nn.ReLU(inplace=inplace)
		 self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
		 self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
		 self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
		 self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
		 self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

		 self._initialize_weights()

	     def forward(self, x):
		 x = self.relu(self.conv1(x))
		 x = self.relu(self.conv2(x))
		 x = self.relu(self.conv3(x))
		 x = self.pixel_shuffle(self.conv4(x))
		 return x

	     def _initialize_weights(self):
		 init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
		 init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
		 init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
		 init.orthogonal_(self.conv4.weight)

	 torch_model = SuperResolutionNet()

      For this tutorial, we will use pre-trained weights provided by the PyTorch team. Note that the model was only partially trained and being used for demonstration purposes.

      .. code-block:: python

	 # Load pretrained model weights
	 model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'

	 # Initialize model with the pretrained weights
	 map_location = lambda storage, loc: storage
	 if torch.cuda.is_available():
	     map_location = None
	 torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

	 # set the model to inference mode
	 torch_model.eval()


      Exporting a model to ONNX in PyTorch works via tracing or
      scripting (read more at `official PyTorch documentation
      <https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting>`_). In
      this tutorial we will export the model using tracing techniques: 

      .. code-block:: python

	 batch_size = 1
	 # Tracing input to the model
	 x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

	 # Export the model
	 torch.onnx.export(
	    torch_model,
	    x,
	    "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
	    export_params=True,  # store the trained parameter weights inside the model file
	    opset_version=10,  # the ONNX version to export the model to
	    do_constant_folding=True,  # whether to execute constant folding for optimization
	    input_names=["input"],  # the model's input names
	    output_names=["output"],  # the model's output names
	    dynamic_axes={
	       "input": {0: "batch_size"},  # variable length axes
	       "output": {0: "batch_size"},
	    },
	 )

      Notice from the arguments of ``torch.onnx.export()``, even though we are exporting the model
      with an input of ``batch_size=1``, the first dimension is still specified as dynamic in ``dynamic_axes``
      parameter. By doing so, the exported model will accept inputs of size ``[batch_size, 1, 224, 224]`` where
      ``batch_size`` can vary among inferences.

      We can now compute the output using ONNX Runtime’s Python APIs:

      .. code-block:: python

	 import onnxruntime

	 ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
	 # compute ONNX Runtime output prediction
	 ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
	 # ONNX Runtime will return a list of outputs
	 ort_outs = ort_session.run(None, ort_inputs)
	 print(ort_outs[0])

   .. tab-item:: TensorFlow
      :sync: tensorflow

      First let's install `tf2onnx <https://github.com/onnx/tensorflow-onnx>`_

      .. code-block:: bash

	 pip install tf2onnx

      For this tutorial we will download a pretrained ResNet-50 model:

      .. code-block:: python

	 import tensorflow as tf
	 from tensorflow.keras.applications.resnet50 import ResNet50

	 model = ResNet50(weights='imagenet')

      Notice that we use ``None`` in `TensorSpec <https://www.tensorflow.org/api_docs/python/tf/TensorSpec>`_ to
      denote the first input dimension as dynamic batch axies, which
      means this dimension can accept any arbitrary input size:

      .. code-block:: python

	 spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
	 onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)


   .. tab-item:: Scikit Learn
      :sync: sklearn

      TODO


Saving ONNX model with BentoML
------------------------------

To quickly save any given ONNX model to BentoML's :ref:`Model
Store<concepts/model:Managing Models>`, use ``onnx.load`` to
load the exported ONNX model back into the Python session,
then call BentoML's :obj:`~bentoml.onnx.save_model()`:


.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. code-block:: python

	 signatures = {
	     "run": {"batchable": True},
	 }
	 bentoml.onnx.save_model("onnx_super_resolution", onnx_model, signatures=signatures)

      which will result:

      .. code-block:: bash

	 Model(tag="onnx_super_resolution:lwqr7ah5ocv3rea3", path="~/bentoml/models/onnx_super_resolution/lwqr7ah5ocv3rea3/")

   .. tab-item:: TensorFlow
      :sync: tensorflow

      .. code-block:: python

	 signatures = {
	     "run": {"batchable": True},
	 }
	 bentoml.onnx.save_model("onnx_resnet50", onnx_model, signatures=signatures)

      which will result:

      .. code-block:: bash

	 Model(tag="onnx_resnet50:zavavxh6w2v3rea3", path="~/bentoml/models/onnx_resnet50/zavavxh6w2v3rea3/")

   .. tab-item:: Scikit Learn
      :sync: sklearn

      TODO

The default signature for :obj:`~bentoml.onnx.save_model()` is set to ``{"run": {"batchable": False}}``.

This means by default, BentoML's :ref:`guides/batching:Adaptive Batching` is disabled when saving ONNX model.
If you want to enable adaptive batching, provide a signature similar to the
aboved example.

Refers to :ref:`concepts/model:Model Signatures` and :ref:`Batching behaviour <concepts/model:Batching>` for more information.


Building a Service for **ONNX**
-------------------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service and APIs>` for how to
   create a prediction service with BentoML.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. code-block:: python

	 import bentoml

	 import numpy as np
	 from PIL import Image as PIL_Image
	 from PIL import ImageOps
	 from bentoml.io import Image

	 runner = bentoml.onnx.get("onnx_super_resolution:latest").to_runner()

	 svc = bentoml.Service("onnx_super_resolution", runners=[runner])

	 @svc.api(input=Image(), output=Image())
	 def sr(img) -> np.ndarray:
	     img = img.resize((224, 224))
	     gray_img = ImageOps.grayscale(img)
	     arr = np.array(gray_img) / 255.0 # convert from 0-255 range to 0.0-1.0 range
	     arr = np.expand_dims(arr, (0, 1)) # add batch_size, color_channel dims
	     sr_arr = runner.run.run(arr)
	     sr_arr = np.squeeze(sr_arr) # remove batch_size, color_channel dims
	     sr_img = PIL_Image.fromarray(np.uint8(sr_arr * 255) , 'L')
	     return sr_img


   .. tab-item:: TensorFlow
      :sync: tensorflow

      .. code-block:: python

	 import bentoml

	 import numpy as np
	 from bentoml.io import Image
	 from bentoml.io import JSON

	 runner = bentoml.onnx.get("onnx_resnet50:latest").to_runner()

	 svc = bentoml.Service("onnx_resnet50", runners=[runner])

	 @svc.api(input=Image(), output=JSON())
	 def predict(img):

	     from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

	     img = img.resize((224, 224))
	     arr = np.array(img)
	     arr = np.expand_dims(arr, axis=0)
	     arr = preprocess_input(arr)
	     preds = runner.run.run(arr)
	     return decode_predictions(preds, top=1)[0]


   .. tab-item:: Scikit Learn
      :sync: sklearn

      TODO

.. note::

   In the aboved example, notice there are two :code:`run` in ``runner.run.run(input_data)`` inside inference code. The distinction between the two ``run`` are as follow:

   1.  The first ``run`` refers  to `onnxruntime.InferenceSession <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/session/inference_session.cc>`_'s ``run`` method, which is ONNX Runtime API to run `inference <https://onnxruntime.ai/docs/api/python/api_summary.html#data-inputs-and-outputs>`_.
   2. The second ``run`` refers to BentoML's runner inference API for invoking a model's signature. In the case of ONNX, it happens to have the same name as the ``InferenceSession`` endpoint.


When constructing a :ref:`bentofile.yaml <concepts/bento:Bento Build
Options>`, there are two ways to include ONNX as a dependency, via
``python`` (if using pip) or ``conda``:

.. tab-set::

   .. tab-item:: python

      .. code-block:: yaml

         python:
         - onnx
	 - onnxruntime

   .. tab-item:: conda

      .. code-block:: yaml

         conda:
           channels:
           - conda-forge
           dependencies:
           - onnx
	   - onnxruntime


Using Runners
-------------

.. seealso::

   :ref:`Runners<concepts/runner:Using Runners>` for more information on what is
   a Runner and how to use it.

To test ONNX Runner locally, access the model via ``get`` and
convert it to a runner object:

.. code-block:: python

   test_input = np.random.randn(2, 1, 244, 244)

   runner = bentoml.onnx.get("super_resolution").to_runner()

   runner.init_local()

   runner.run.run(test_input)

.. note::

   You don't need to cast your input ndarray to ``np.float32`` for
   runner input.

Similar to ``load_model``, you can customize ``providers`` and ``session_options`` when creating a runner:

.. code-block:: python

   providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

   bento_model = bentoml.onnx.get("onnx_super_resolution")

   runner = bento_model.with_options(providers=providers).to_runner()

   runner.init_local()


Loading an ONNX model with BentoML for local testing
----------------------------------------------------

Use ``load_model`` to verify that the saved model can be loaded properly:

.. code-block:: python

   ort_session = bentoml.onnx.load_model("onnx_super_resolution")

.. note::

   BentoML will load an ONNX model back as an
   ``onnxruntime.InferenceSession`` object which is ready to do
   inference

.. code-block:: python

   test_input = np.random.randn(2, 1, 244, 244) # can accept arbitrary batch size
   ort_session.run(None, {"input": test_input.astype(np.float32)})

.. note::

   In the above snippet, we need explicitly convert input ndarray to
   float32 since ``onnxruntime.InferenceSession`` expects only single floats. 

   However, BentoML will automatically cast the input data automatically via Runners.


Dynamic Batch Size
------------------

.. seealso::

   :ref:`guides/batching:Adaptive Batching`: a general introduction to adaptive batching in BentoML.

When :ref:`guides/batching:Adaptive Batching` is enabled, the exported
ONNX model is **REQUIRED** to accept dynamic batch size. 

Therefore, dynamic batch axes needs to be specified when the model is exported to the ONNX format.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      For PyTorch models, you can achieve this by specifying ``dynamic_axes``
      when using `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_

      .. code-block:: python

	 torch.onnx.export(
	    torch_model,
	    x,
	    "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
	    export_params=True,  # store the trained parameter weights inside the model file
	    opset_version=10,  # the ONNX version to export the model to
	    do_constant_folding=True,  # whether to execute constant folding for optimization
	    input_names=["input"],  # the model's input names
	    output_names=["output"],  # the model's output names
	    dynamic_axes={
	       "input": {0: "batch_size"},  # variable length axes
	       "output": {0: "batch_size"},
	    },
	 )

   .. tab-item:: TensorFlow
      :sync: tensorflow

      For TensorFlow models, you can achieve this by using ``None`` to denote
      a dynamic batch axis in `TensorSpec
      <https://www.tensorflow.org/api_docs/python/tf/TensorSpec>`_ when
      through either ``tf2onnx.convert.from_keras`` or ``tf2onnx.convert.from_function``

      .. code-block:: python

	 spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),) # batch_axis = 0
	 model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)


   .. tab-item:: Scikit Learn
      :sync: sklearn

      TODO

Default Execution Providers Settings
------------------------------------

When a CUDA-compatible GPU is available, BentoML runner will use ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` as the de facto execution providers.

Otherwise, Runner will use ``["CPUExecutionProvider"]`` as the default providers.

If ``onnxruntime-gpu`` is installed, using ``TensorrtExecutionProvider`` may improve inference runtime. You can
override the default setting using ``with_options`` when creating a runner:

.. code-block:: python

   providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

   bento_model = bentoml.onnx.get("onnx_super_resolution")

   runner = bento_model.with_options(providers=providers).to_runner()

.. seealso::

   `Execution Providers' documentation <https://onnxruntime.ai/docs/execution-providers/>`_
