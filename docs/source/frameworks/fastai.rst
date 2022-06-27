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

We can then test out prediction:

.. code-block:: python

   learner.predict("I really liked that movie!")

   # output:
   # ('pos', TensorText(1), TensorText([0.1216, 0.8784]))


Saving a learner with BentoML
-----------------------------

   :bdg-warning:`Warning:` `Learner <https://docs.fast.ai/learner.html#Learner>`_ instance is required to save with BentoML.
   This is a design choice to preserve functionalities provided by fastai.

.. seealso::

   If you want to use the PyTorch model components of fastai ``Learner``s with BentoML, refer to the :ref:`PyTorch Framework Guide<frameworks/pytorch:PyTorch>`.

   To get the PyTorch model, access it via ``learner.model``:

   .. code-block:: python

      import bentoml

      bentoml.pytorch.save_model("my_pytorch_model", learner.model)

To quickly save the trained learner, use ``save_model``:

.. code-block:: python

   bentoml.fastai.save_model("fastai_sentiment", learner)

   # output:
   # Model(tag="fastai_sentiment:5bakmghqpk4z3gxi", path="~/bentoml/models/fastai_sentiment/5bakmghqpk4z3gxi/")

.. seealso::
   It is also possible to **use pre-trained models** directly with BentoML, without
   saving it to the model store first. Check out the
   :ref:`Custom Runner <concepts/runner:Custom Runner>` example to learn more.

.. tip::

   If you have an existing model saved to file on disk, you will need to load the model
   in a python session first and then use BentoML's framework specific
   :code:`save_model` method to put it into the BentoML model store.

   We recommend **always save the model with BentoML as soon as it finished training and
   validation**. By putting the :code:`save_model` call to the end of your training
   pipeline, all your finalized models can be managed in one place and ready for
   inference.

Optionally, you may attach custom labels, metadata, or :code:`custom_objects` to be
saved alongside your model in the model store, e.g.:

.. code:: python

    bentoml.fastai.save_model(
        "fastai_sentiment",       # model name in the local model store
        learner,                  # model instance being saved
        labels={                  # user-defined labels for managing models in Yatai
            "owner": "nlp_team",
            "stage": "dev",
        },
        metadata={                # user-defined additional metadata
            "dataset_version": "20210820",
        },
        custom_objects={          # save additional user-defined python objects
            "tokenizer": tokenizer,
        }
    )

- **labels**: user-defined labels for managing models, e.g. :code:`team=nlp`, :code:`stage=dev`.
- **metadata**: user-defined metadata for storing model training context information or model evaluation metrics, e.g. dataset version, training parameters, model scores.
- **custom_objects**: user-defined additional python objects, e.g. a tokenizer instance, preprocessor function, model configuration json, serialized with cloudpickle. Custom objects will be serialized with `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_.

.. note::

   By default, BentoML uses ``cloudpickle`` for learner serialization. The key difference from Python's ``pickle`` is that 
   ``cloudpickle`` has the capability to serialize functions and so it can directly serialize members of the object without reference to its type.

   If the :func:`save_model` method failed while saving a given learner, your learner may contain a :obj:`~fastai.callback.core.Callback` that is not picklable.
   All FastAI callbacks are stateful, which makes some of them not picklable. Use :func:`Learner.remove_cbs` to remove unpicklable callbacks.

   BentoML by default removes some known callback that have issues with serialization, such as ``ParamScheduler``. 

.. tip::

   We found that some of given callback are only useful during training phase, not during serving. Therefore, removing callback will not affect inference results.


Loading a learner with BentoML
------------------------------

.. warning::

   We recommend users to to use ``load_model`` inside a :obj:`bentoml.Service`.

   You should always use ``bentoml.models.get("model:tag").to_runner()`` to get
   a :obj:`bentoml.Runner` instead. See also :ref:`Runners <concepts/runner:Using Runners>`_ for more information.


To load the learner back to memory, use ``load_model``:

.. code-block:: python

   learner = bentoml.fastai.load_model("fastai_sentiment")

Proceed to then test the learner with prediction inputs:

.. code-block:: python

   learner.predict("I really liked that movie!")

.. admonition:: About the behaviour of :code:`load_model()`

   Since fastai doesn't provide a good support for GPU during inference, BentoML
   by default will only support CPU inference for fastai. If you want to use
   GPU, you should get the ``PyTorch`` model from ``learner.model`` and then use
   ``bentoml.pytorch`` instead.

   Additionallly, if the model uses ``mixed_precision``, then the loaded model will also be converted to FP32.
   Learn more about `mixed precision <https://docs.fast.ai/callback.fp16.html>`_.


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
-----------------------------

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
