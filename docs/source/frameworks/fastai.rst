=======
fast.ai
=======


Preface
-------

In this tutorial, we will show how to train a model for sentiment analysis with fastai, then we will use BentoML to save and create a prediction service.

We will use the IMDb dataset from the paper |stanford_link|_, which contains several thousand movie reviews, for this tutorial.

.. note::

   BentoML requires ``fastai>=2.0`` to be installed. We will not provide support
   for fastai older than version 2 as version 2 includes a lot breaking changes
   comparing to version 1.


.. seealso::

   This tutorial is extracted from `Transfer Learning with text <https://docs.fast.ai/tutorial.text.html#The-ULMFiT-approach>`_ from fastai.

Fine tuning the language model
------------------------------

First, import all required components from fastai

.. code-block:: python

   from fastai.basics import URLs
   from fastai.metrics import accuracy
   from fastai.text.data import DataBlock
   from fastai.text.data import TextBlock
   from fastai.text.data import untar_data
   from fastai.text.data import CategoryBlock
   from fastai.text.models import AWD_LSTM
   from fastai.text.learner import text_classifier_learner
   from fastai.data.transforms import parent_label
   from fastai.data.transforms import get_text_files
   from fastai.data.transforms import GrandparentSplitter

We can then download the data and decompress it with ``untar_data``:

.. code-block:: python

   path = untar_data(URLs.IMDB)

We can use the data block API to get our data in a `DataLoaders <https://docs.fast.ai/data.core.html#DataLoaders>`_. 

The data follows an ImageNet-style organization, in the train folder, we have two subfolders, `pos` and `neg` (for positive reviews and negative reviews).

.. code-block:: python

   imdb = DataBlock(
       blocks=(TextBlock.from_folder(path), CategoryBlock),
       get_items=get_text_files,
       get_y=parent_label,
       splitter=GrandparentSplitter(valid_name="test"),
   )
   dls = imdb.dataloaders(path)

.. note::

   The data block API is an advanced feature from fastai, if you prefer to use other methods
   refers to the `main tutorial <https://docs.fast.ai/tutorial.text.html#Using-the-high-level-API>`_
   for more information.

Then, we can define a `Learner <https://docs.fast.ai/learner.html#Learner>`_ suitable for text classification in one line:

.. code-block:: python

   learner = text_classifier_learner(
        dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy
    )

We use the `AWD LSTM <https://arxiv.org/abs/1708.02182>`_ architecture, *drop_mult* is a parameter that controls the magnitude of all dropouts in that model, and we use `accuracy <https://docs.fast.ai/metrics.html#accuracy>`_ to track down how well we are doing. We can then fine-tune our pretrained model:

.. code-block:: python

   learner.fine_tune(4, 1e-2)

.. code-block:: bash

   epoch     train_loss  valid_loss  accuracy  time
   0         0.453252    0.395130    0.822080  36:45

We can then test out prediction:

.. code-block:: python

   learner.predict("I really liked that movie!")

.. code-block:: bash

   ('pos', TensorText(1), TensorText([0.1216, 0.8784]))


Saving a learner with BentoML
-----------------------------

   :bdg-warning:`Warning:` ``Learner`` instance is required to save with BentoML.
   This is a design choice to preserve functionalities provided by fastai.

.. seealso::

   Refers to :ref:`PyTorch Framework Guide<frameworks/pytorch:PyTorch>` for more information if one wants to use PyTorch model components of ``Learner`` with BentoML.

   To get the PyTorch model, access it via ``learner.model``:

   .. code-block:: python

      import bentoml

      bentoml.pytorch.save_model("my_pytorch_model", learner.model)

To quickly save the trained learner, use ``save_model``:

.. code-block:: python

   bentoml.fastai.save_model("fastai_sentiment", learner)

.. code-block:: bash

   Model(tag="fastai_sentiment:5bakmghqpk4z3gxi", path="~/bentoml/models/fastai_sentiment/5bakmghqpk4z3gxi/")

In addition to :ref:`general options <concepts/model:Save A Trained Model>`
provided by :code:`save_model`, you can optionally provide a different ``pickle_module``
for serializing the model.

.. code-block:: python

   import pickle
   
   bentoml.fastai.save_model("fastai_sentiment", learner, pickle_module=pickle)

.. note::

   By default, BentoML uses ``cloudpickle`` for serialization. The key difference from Python's ``pickle`` is that 
   ``cloudpickle`` has the capability to serialize functions and so it can directly serialize members of the object without reference to its type.

   :bdg-primary:`Our Recommendation:` ``cloudpickle`` *should be used for most cases.*

.. admonition:: about :code:`save_model()` behaviour

   BentoML also tries to remove some known callback that have issues with
   serialization, such as ``ParamScheduler``. 

   We found that some of given callback are only useful during training phase, not during serving, thus 
   not affecting inference results.


Loading a learner with BentoML
------------------------------

To load the learner back to memory, use ``load_model``:

.. code-block:: python

   learner = bentoml.fastai.load_model("fastai_sentiment")

Proceed to then test the learner with prediction inputs:

.. code-block:: python

   learner.predict("I really liked that movie!")

In addition to :ref:`general options <concepts/model:Retrieve a saved model>`
provided by :code:`load_model`, you can also provide ``cpu`` to enforce loading
the learner on CPU.

.. code-block:: python

   learner = bentoml.fastai.load_model("fastai_sentiment", cpu=False)


.. admonition:: About the behaviour of :code:`cpu=True`

   fastai will determine which devices to use (GPU or CPU) via ``cpu``. The
   results will then be passed down to ``map_location`` of ``torch.load``.
   Refers to `PyTorch's documentation <https://pytorch.org/docs/stable/generated/torch.load.html#torch-load>`_
   for more information.

   Additionallly, if the model uses ``mixed_precision``, then the loaded model will also be converted to FP32.
   Learn more about `mixed precision <https://docs.fast.ai/callback.fp16.html>`_.


   :bdg-primary:`Remarks:` BentoML are currently only providing CPU supports for fastai.


Using Runners
-------------

.. seealso::

   :ref:`Runners' documentation<concepts/runner:Using Runners>` on Runners' concept and its usage.

.. seealso::

   :ref:`Specifying Runner Resources<concepts/runner:Specifying Required Resources>` on providing options for Runners.


To use fastai runner locally, access the model via ``get`` and convert it to
a runner:

.. code-block:: python

   runner = bentoml.fastai.get("fastai_sentiment").to_runner()

   runner.init_local()

   runner.predict.run("I really liked that movie!")

.. note::

   Since fastai contains different implementation for different ``Learner``
   type (Tabular, Text, Vision, etc.), users need to be responsible for
   processing and converting model inputs to corresponding format.

.. admonition:: About adaptive batching in fastai 

   fastai doesn't have support for multiple inputs, hence adaptive batching
   is disabled for fastai. Refers to :ref:`guides/batching:Adaptive Batching` for more information.

Building a Service for fastai
---------------------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service and APIs>` for how to
   create a prediction service with BentoML.

When constructing a :ref:`bentofile.yaml <concepts/bento:Bento Build Options>`,
there are two ways to include fastai as a dependency, via ``python`` or
``conda``:

.. tab-set::

   .. tab-item:: python

      .. code-block:: yaml

         python:
         - fastai

   .. tab-item:: conda

      .. code-block:: yaml

         conda:
           channels:
           - fastchan
           dependencies:
           - fastai


.. note::

   You can find more examples for fastai in our `gallery <https://github.com/bentoml/gallery>`_ repo.


.. _stanford_link: https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf

.. |stanford_link| replace:: *Learning Word Vectors for Sentiment Analysis*
