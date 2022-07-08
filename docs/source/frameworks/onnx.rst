====
ONNX
====


Preface
-------

ONNX is an open format built to represent machine learning models. ONNX provides `high interoperability <https://onnx.ai/supported-tools.html#buildModel>`_  among various frameworks, as well as enable machine learning practitioners to maximize models' performance across `different hardware <https://onnx.ai/supported-tools.html#deployModel>`_.

Due to its high interoperability among frameworks, we recommend you to check out the framework integration with ONNX as it will contain specific recommendation and requirements for that given framework.


.. tab:: PyTorch

   - `Quick tutorial about exporting a model from PyTorch to ONNX <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>`_ from official PyTorch documentation.
   - `torch.onnx <https://pytorch.org/docs/stable/onnx.html>`_ from official PyTorch documentation. Pay special attention to section **Avoiding Pitfalls**, **Limitations** and **Frequently Asked Questions**.

.. tab:: TensorFlow

   - `tensorflow-onnx (tf2onnx) <https://github.com/onnx/tensorflow-onnx>`_ documentation.

.. tab:: Scikit Learn

   TODO

Converting model frameworks to ONNX format
-----------------------------------------------

.. note::

   BentoML currently only support `ONNX Runtime
   <https://onnxruntime.ai>`_ as ONNX backend. BentoML requires either
   ``onnxruntime>=1.9`` or ``onnxruntime-gpu>=1.9`` to be installed.

.. tab:: PyTorch

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

   For this tutorial, we will download some pre-trained weights. Note
   that this model was not trained fully for good accuracy and is used
   here for demonstration purposes only.

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


   Exporting a model to onnx in PyTorch works via tracing or
   scripting. In this tutorial we will export a model using
   tracing. Note how we export the model with an input of
   ``batch_size=1``, but then specify the first dimension as dynamic
   in the ``dynamic_axes`` parameter in ``torch.onnx.export()``. The
   exported model will thus accept inputs of size ``[batch_size, 1,
   224, 224]`` where ``batch_size`` can vary among each inference.

   .. code-block:: python

      batch_size = 1 # can be any number
      # Tracing input to the model
      x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

      # Export the model
      torch.onnx.export(torch_model,
			x,
			"super_resolution.onnx",   # where to save the model (can be a file or file-like object)
			export_params=True,        # store the trained parameter weights inside the model file
			opset_version=10,          # the ONNX version to export the model to
			do_constant_folding=True,  # whether to execute constant folding for optimization
			input_names=['input'],   # the model's input names
			output_names=['output'], # the model's output names
			dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
				      'output' : {0 : 'batch_size'}})

   Now we can compute the output using ONNX Runtime’s Python APIs:

   .. code-block:: python

      import onnxruntime

      ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
      # compute ONNX Runtime output prediction
      ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
      # ONNX Runtime will return a list of outputs
      ort_outs = ort_session.run(None, ort_inputs)
      print(ort_outs[0])

.. tab:: TensorFlow

   First let's install `tf2onnx <https://github.com/onnx/tensorflow-onnx>`_

   .. code-block:: bash

      pip install tf2onnx

   For this tutorial we will download a pretrained ResNet-50 model:

   .. code-block:: python

      import tensorflow as tf
      from tensorflow.keras.applications.resnet50 import ResNet50

      model = ResNet50(weights='imagenet')

   Then we can export the model to ONNX format. Notice that we use
   ``None`` in `TensorSpec
   <https://www.tensorflow.org/api_docs/python/tf/TensorSpec>`_ to
   denote the first input dimension as dynamic batch axies, which
   means this dimension can accept arbitrary input size.

   .. code-block:: python

      spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
      onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)


.. tab:: Scikit Learn

   TODO


Saving ONNX model with BentoML
-----------------------------

To quickly save an ONNX model to BentoML's :ref:`Model
Store<concepts/model:Managing Models>`, first use ``onnx.load`` to
load the exported ONNX model back into ``onnx.ModelProto`` object,
then call BentoML's ``save_model``:


.. tab:: PyTorch

   .. code-block:: python

      signatures = {
	  "run": {"batchable": True},
      }
      bentoml.onnx.save_model("onnx_super_resolution", onnx_model, signatures=signatures)

   which will result:

   .. code-block:: bash

      Model(tag="onnx_super_resolution:lwqr7ah5ocv3rea3", path="~/bentoml/models/onnx_super_resolution/lwqr7ah5ocv3rea3/")

.. tab:: TensorFlow

   .. code-block:: python

      signatures = {
	  "run": {"batchable": True},
      }
      bentoml.onnx.save_model("onnx_resnet50", onnx_model, signatures=signatures)

   which will result:

   .. code-block:: bash

      Model(tag="onnx_resnet50:zavavxh6w2v3rea3", path="~/bentoml/models/onnx_resnet50/zavavxh6w2v3rea3/")

.. tab:: Scikit Learn

   TODO

.. note::

   ``save_model`` will use ``{"run": {"batchable": False}}`` as
   default signatures if ``signatures`` is not provided. Set
   ``batchable`` to ``False`` will disable BentoML's
   :ref:`guides/batching:Adaptive Batching` functionality. That's why
   we provide our own signatures here. Read more about :ref:`Model
   Signatures <concepts/model:Model Signatures>` and :ref:`Batch Input
   <concepts/model:Batching>`

.. seealso::

   ``save_model`` also has some :ref:`general options
   <concepts/model:Save A Trained Model>` for functionalities like
   saving metadata and custom objects.


Building a Service for **ONNX**
-------------------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service and APIs>` for how to
   create a prediction service with BentoML.

.. tab:: PyTorch

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


.. tab:: TensorFlow

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


.. tab:: Scikit Learn

   TODO

.. note::

   In above codes we use ``runner.run.run(input_data)`` to do
   inference. The first ``run`` is referring to
   ``onnxruntime.InferenceSession``'s ``run`` method, while the second
   ``run`` is BentoML's naming convention for doing runner inference
   for a model method. For example, for a Keras model with ``predict``
   method, we will call ``runner.predict.run(input_data)``.


When constructing a :ref:`bentofile.yaml <concepts/bento:Bento Build Options>`,
there are two ways to include ONNX as a dependency, via ``python`` or
``conda``:

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

To use ``onnx`` runner locally, access the model via ``get`` and
convert it to a runner:

.. code-block:: python

   test_input = np.random.randn(2, 1, 244, 244)

   runner = bentoml.onnx.get("super_resolution").to_runner()

   runner.init_local()

   runner.run.run(test_input)

.. note::

   You don't need to cast your input ndarray to ``np.float32`` for
   runner input

Like ``load_model``, you can customize ``providers`` and
``session_options`` when you create a runner:

.. code-block:: python

   providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

   runner = bentoml.onnx.get("onnx_super_resolution").with_options(providers=providers).to_runner()

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


Then we can do some test inference:

.. code-block:: python

   test_input = np.random.randn(2, 1, 244, 244) # can accept arbitrary batch size
   ort_session.run(None, {"input": test_input.astype(np.float32)})

.. note::

   In above codes we need explicitly to convert input ndarray to
   float32 because ``onnxruntime.InferenceSession`` only expects
   single floats. When using BentoML runner, it will automatically
   cast input data to this type


Dynamic Batch Size
------------------

When enabling :ref:`guides/batching:Adaptive Batching`, the exported
ONNX model need to accept dynamic batch size. Hence the dynamic batch
axes need to be specified when the mode is exported in ONNX format.

.. tab:: PyTorch

   For PyTorch models, you can do that by specifying ``dynamic_axes``
   when using `torch.onnx.export
   <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_

   .. code-block:: python

      torch.onnx.export(torch_model,
			x,
			"super_resolution.onnx",   # where to save the model (can be a file or file-like object)
			export_params=True,        # store the trained parameter weights inside the model file
			opset_version=10,          # the ONNX version to export the model to
			do_constant_folding=True,  # whether to execute constant folding for optimization
			input_names=['input'],   # the model's input names
			output_names=['output'], # the model's output names
			dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
				      'output' : {0 : 'batch_size'}})

.. tab:: TensorFlow

   For TensorFlow models, you can do that by using ``None`` to denote
   a dynamic batch axis in `TensorSpec
   <https://www.tensorflow.org/api_docs/python/tf/TensorSpec>`_ when
   using ``tf2onnx.convert.from_keras`` or
   ``tf2onnx.convert.from_function``

   .. code-block:: python

      spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),) # batch_axis = 0
      model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)


.. tab:: Scikit Learn

   TODO

Default Execution Providers Settings
------------------------------------

* When a CUDA compatible GPU is available, BentoML runner will use ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` as the default Execution Providers.
* When CUDA compatible GPU is not available, BentoML runner will use
  ``["CPUExecutionProvider"]`` as the default Execution Providers.

If dependencies are installed, using other Execution Providers like
``TensorrtExecutionProvider`` may increase the performance. You can
override the default setting using ``with_options`` when creating the
runner:

.. code-block:: python

   providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

   runner = bentoml.onnx.get("onnx_super_resolution").with_options(providers=providers).to_runner()

You can read more about Execution Providers at ONNX Runtime's
`official documentation
<https://onnxruntime.ai/docs/execution-providers/>`_
