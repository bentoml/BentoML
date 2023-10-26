=======
EasyOCR
=======

EasyOCR is a ready-to-use OCR with 80+ supported languages. It helps you to quickly convert and transcribe text from images. This guide provides an overiew of using `EasyOCR <https://www.jaided.ai/easyocr/>`_ with BentoML.

Compatibility
-------------

BentoML has been validated to work with EasyOCR version 1.6.2 and higher.

Save/Load a EasyOCR Reader with BentoML
---------------------------------------

First, create a reader instance with the `language codes <https://www.jaided.ai/easyocr/>`_ for your usecase.

.. code-block:: python

   import easyocr

    reader = easyocr.Reader(['en'])

Save this reader instance using :obj:`~bentoml.easyocr.save_model()` to save this to the BentoML model store

.. code-block:: python

   import bentoml

   bento_model = bentoml.easyocr.save_model('en-reader', reader)

To verify that the saved model is working, load it back with :obj:`~bentoml.easyocr.load_model()`:

.. code-block:: python

   loaded_model = bentoml.easyocr.load_model('en-reader')

   rs = loaded_model.readtext('image.jpg')

.. note:: GPU behaviour

   GPU can be passed through ``easyocr.Reader`` constructor as ``gpu=True``. This means in order to use GPU, the reader instance must be created with a machine with GPU before saving it to BentoML.

Building a Service
------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service APIs>`: more information on creating a
   prediction service with BentoML.

Create a ``service.py`` file separate from your training code that will be used to define the
BentoML service:

.. code-block:: python

   import bentoml
   import PIL.Image
   import numpy as np

   # create a runner from the saved Booster
   runner = bentoml.easyocr.get("en-reader").to_runner()

   # create a BentoML service
   svc = bentoml.Service("ocr", runners=[runner])

   # define a new endpoint on the BentoML service
   @svc.api(input=bentoml.io.Image(), output=bentoml.io.JSON())
   async def transcript_text(input: PIL.Image.Image) -> list:
       # use 'runner.predict.run(input)' instead of 'booster.predict'
       return await runner.readtext.async_run(np.asarray(input))

Take note of the name of the service (``svc`` in this example) and the name of the file.

You should also have a ``bentofile.yaml`` alongside the service file that specifies that
information, as well as the fact that it depends on XGBoost. This can be done using either
``python`` (if using pip), or ``conda``:

.. tab-set::

   .. tab-item:: pip

      .. code-block:: yaml

         service: "service:svc"
         python:
           packages:
              - easyocr
              - bentoml

   .. tab-item:: conda

      .. code-block:: yaml

         service: "service:svc"
         conda:
           channels:
           - conda-forge
           dependencies:
           - easyocr

Using Runners
~~~~~~~~~~~~~

.. seealso::

   :ref:`concepts/runner:Using Runners`: a general introduction to the Runner concept and its usage.

A runner for a Reader is created like so:

.. code-block:: python

   bentoml.easyocr.get("model_name:model_version").to_runner()

``runner.readtext.run`` is generally a drop-in replacement for ``reader.readtext``.

Runners must to be initialized in order for their ``run`` methods to work. This is done by BentoML
internally when you serve a bento with ``bentoml serve``. See the :ref:`runner debugging guide
<concepts/service:Debugging Runners>` for more information about initializing runners locally.


.. currentmodule:: bentoml.easyocr
