=====
Keras
=====

.. note::

   Both ``bentoml.keras`` and ``bentoml.tensorflow`` support Keras
   models. ``bentoml.keras`` utilizes the native model format and
   will give a better development experience to users who are more
   familiar with Keras models. However, the native model format of Keras is
   not optimized for production inference. There are `known reports
   <https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue+sort%3Aupdated-desc+keras+memory+leak>`_
   of memory leaks during serving time at the time of BentoML 1.0
   release, so ``bentoml.tensorflow`` is recommended in production
   environments. You can read :doc:`bentoml.tensorflow
   </frameworks/tensorflow>` documentation for more information.

   You can also convert a Keras model to ONNX model and use
   ``bentoml.onnx`` to serve in production. Refer
   :doc:`bentoml.onnx documentation </frameworks/onnx>` and
   `tensorflow-onnx (tf2onnx) documentation
   <https://github.com/onnx/tensorflow-onnx>`_ for more information.


Compatibility
-------------

BentoML requires TensorFlow version **2.7.3** or higher to be installed.


Saving a Keras Model
--------------------

The following example loads a pre-trained ResNet50 model.

.. code-block:: python

   import tensorflow as tf
   from tensorflow.keras.applications.resnet50 import ResNet50

   # Use pre-trained ResNet50 weights
   model = ResNet50(weights='imagenet')

   # try a sample input with created model
   from tensorflow.keras.preprocessing import image
   from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

   img_path = 'ade20k.jpg'

   img = image.load_img(img_path, target_size=(224, 224))

   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)

   preds = model.predict(x)
   print('Keras Predicted:', decode_predictions(preds, top=3)[0])

   # output:
   # Keras Predicted: [('n04285008', 'sports_car', 0.3447785)]


After the Keras model is ready, use :obj:`~bentoml.keras.save_model`
to save the model instance to BentoML model store.

.. code-block:: python

   bentoml.keras.save_model("keras_resnet50", model)


Keras model can be loaded with :obj:`~bentoml.keras.load_model` to 
verify that the saved model can be loaded properly.

.. code-block:: python

   model = bentoml.keras.load_model("keras_resnet50:latest")

   print(decode_predictions(model.predict(x)))


Building a Service using Keras
------------------------------

.. seealso::

   See :ref:`Building a Service <concepts/service:Service and APIs>` for more 
   information on creating a prediction service with BentoML.

The following service example creates a ``predict`` API endpoint that accepts an image as input 
and return JSON data as output. Within the API function, Keras model runner created from the 
previously saved ResNet50 model is used for inference.

.. code-block:: python

   import bentoml

   import numpy as np
   from bentoml.io import Image
   from bentoml.io import JSON

   runner = bentoml.keras.get("keras_resnet50:latest").to_runner()

   svc = bentoml.Service("keras_resnet50", runners=[runner])

   @svc.api(input=Image(), output=JSON())
   async def predict(img):

       from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

       img = img.resize((224, 224))
       arr = np.array(img)
       arr = np.expand_dims(arr, axis=0)
       arr = preprocess_input(arr)
       preds = await runner.async_run(arr)
       return decode_predictions(preds, top=1)[0]


When constructing a :ref:`bentofile.yaml <concepts/bento:Bento Build
Options>`, there are two ways to include Keras as a dependency, via
``python`` (if using pip) or ``conda``:

.. tab-set::

   .. tab-item:: python

      .. code-block:: yaml

	 python:
	   packages:
	     - tensorflow

   .. tab-item:: conda

      .. code-block:: yaml

         conda:
           channels:
           - conda-forge
           dependencies:
           - tensorflow


Using Runners
-------------

.. seealso::

   See :ref:`concepts/runner:Using Runners` doc for a general introduction to the Runner concept and its usage.


``runner.predict.run`` is generally a drop-in replacement for
``model.predict`` for executing the prediction in the model
runner. When ``predict`` is the only prediction method exposed by
runner model, you can just use ``runner.run`` instead of
``runner.predict.run``
