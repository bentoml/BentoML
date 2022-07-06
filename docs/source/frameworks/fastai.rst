=======
fast.ai
=======

.. note::

   BentoML requires fastai version 2 or higher to be installed. We do not currently support version 1.

fastai is a deep learning library which provides both high-level components for practioners to get SOTA results in standard deep learning domains, and low-level components
for researchers to build new approaches. The abstractions can be achieved by leveraging the flexibility of the PyTorch library and its ecosystem (PyTorch Lightning, Catalyst, etc.)

fastai also provides a great sets of `documentation <docs.fast.ai>`_ and
`migration guides <https://docs.fast.ai/#Migrating-from-other-libraries>`_ from
PyTorch-related library.


Fine tuning the language model
------------------------------

.. seealso::

   This section is based heavily on `Transfer Learning with text <https://docs.fast.ai/tutorial.text.html#The-ULMFiT-approach>`_ from fastai.

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

   # Download IMDB dataset
   path = untar_data(URLs.IMDB)

   # Create IMDB DataBlock
   imdb = DataBlock(
       blocks=(TextBlock.from_folder(path), CategoryBlock),
       get_items=get_text_files,
       get_y=parent_label,
       splitter=GrandparentSplitter(valid_name="test"),
   )
   dls = imdb.dataloaders(path)

   # define a Learner object
   learner = text_classifier_learner(
        dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy
    )

   # quickly fine tune the model
   learner.fine_tune(4, 1e-2)

   # output:
   # epoch     train_loss  valid_loss  accuracy  time
   # 0         0.453252    0.395130    0.822080  36:45

   learner.predict("I really liked that movie!")

   # output:
   # ('pos', TensorText(1), TensorText([0.1216, 0.8784]))


Saving a learner with BentoML
-----------------------------

   :bdg-warning:`Warning:` `Learner <https://docs.fast.ai/learner.html#Learner>`_ instance is required to save with BentoML.
   This is a design choice to preserve functionalities provided by fastai.

To save the trained learner, use ``save_model``:

.. code-block:: python

   bentoml.fastai.save_model("fastai_sentiment", learner)

   # output:
   # Model(tag="fastai_sentiment:5bakmghqpk4z3gxi", path="~/bentoml/models/fastai_sentiment/5bakmghqpk4z3gxi/")

.. note::

   If you want to use the PyTorch model components of fastai ``Learner``s with BentoML, refer to the :ref:`PyTorch Framework Guide<frameworks/pytorch:PyTorch>`.

   To get the PyTorch model, access it via ``learner.model``:

   .. code-block:: python

      import bentoml

      bentoml.pytorch.save_model("my_pytorch_model", learner.model)


Loading a learner with BentoML
------------------------------

To load the learner back to memory, use ``load_model``:

.. code-block:: python

   learner = bentoml.fastai.load_model("fastai_sentiment:latest")

You can then proceed to test the learner with prediction inputs:

.. code-block:: python

   learner.predict("I really liked that movie!")

.. warning::

   We recommend users to to use ``load_model`` inside a :obj:`bentoml.Service`.

   You should always use ``bentoml.models.get("model:tag").to_runner()`` to get
   a :obj:`bentoml.Runner` instead. See also :ref:`Runners <concepts/runner:Using Runners>` for more information.


.. admonition:: About the behaviour of :code:`load_model()`

   Since fastai doesn't provide a good support for GPU during inference, BentoML
   by default will only support CPU inference for fastai. If you want to use
   GPU, you should get the ``PyTorch`` model from ``learner.model`` and then use
   ``bentoml.pytorch`` instead.

   Additionally, if the model uses ``mixed_precision``, then the loaded model will also be converted to FP32.
   See `mixed precision <https://docs.fast.ai/callback.fp16.html>`_ to learn more about mixed precision.


Using Runners
-------------

.. seealso::

   :ref:`The general Runner documentation<concepts/runner:Using Runners>`: general information about the Runner concept and their usage.

   :ref:`Specifying Runner Resources<concepts/runner:Specifying Required Resources>`: more information about Runner options.

.. tip::

   ``runner.predict.run`` should generally be a drop-in replacement for ``learner.predict`` regardless of your learner type.

   A fastai :~obj:`bentoml.Runner` is a wrapper around a fastai :~obj:`Learner`
   object. This means that a fastai runner will receive the same inputs type as
   the given learner.

   i.e: `Tabular runner <https://docs.fast.ai/tabular.learner.html>`_ will
   accept a :obj:`DataFrame` input. Text runner will accept a :obj:`str` input,
   etc.

   Note that fast.ai doesn't provide support for batched inputs and multiple
   input types, hence :code:`batchable` should always be set to ``False``.



Building a Service for fastai
-----------------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service and APIs>`: more information on creating a prediction service with BentoML.

.. code-block:: python

   import bentoml

   import numpy as np

   from bentoml.io import Text
   from bentoml.io import NumpyNdarray

   runner = bentoml.fastai.get("fastai_sentiment:latest").to_runner()

   fastsvc = bentoml.Service("fast_sentiment", runners=[runner])


   @fastsvc.api(input=Text(), output=NumpyNdarray())
   def classify_text(text: str) -> np.ndarray:
      # returns sentiment score of a given text
      res = runner.predict.run(text)
      return np.asarray(res[-1])


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
