============
Transformers
============

`🤗 Transformers <https://huggingface.co/docs/transformers/main/en/index>`_ is a popular open-source library for natural language processing,
providing pre-trained models and tools for building, training, and deploying custom language models. It offers support for a wide
range of transformer-based architectures, access to pre-trained models for various NLP tasks, and the ability to fine-tune pre-trained models on
specific tasks. BentoML provides native support for serving and deploying models trained from 
Transformers.

Compatibility 
-------------

BentoML requires Transformers version 4 or above. For other versions of Transformers, consider using a 
:ref:`concepts/runner:Custom Runner`.

When constructing a :ref:`bentofile.yaml <concepts/bento:Bento Build Options>`, include ``transformers`` and the machine learning 
framework of the model, e.g. ``pytorch``, ``tensorflow``, or ``jax``.

.. tab-set::

   .. tab-item:: PyTorch

      .. code-block:: yaml
         :caption: `bentofile.yaml`

         service: "service.py:svc"
         labels:
         owner: bentoml-team
         project: gallery
         include:
         - "*.py"
         python:
           packages:
           - transformers
           - torch

   .. tab-item:: TensorFlow

      .. code-block:: yaml
          :caption: `bentofile.yaml`

          service: "service.py:svc"
          labels:
          owner: bentoml-team
          project: gallery
          include:
          - "*.py"
          python:
            packages:
            - transformers
            - tensorflow

Pre-Trained Models
------------------

Transformers provides pre-trained models for a wide range of tasks, including text classification, question answering, language translation,
and text generation. The pre-trained models have been trained on large amounts of data and are designed to be fine-tuned on specific downstream
tasks. Fine-tuning pretrained models is a highly effective practice that enables users to reduce computation costs while adapting state-of-the-art
models to their specific domain dataset. To facilitate this process, Transformers provides a diverse range of libraries specifically designed for
fine-tuning pretrained models. To learn more, refer to the Transformers guide on 
`fine-tuning pretrained models <https://huggingface.co/docs/transformers/main/en/training>`_.

.. tip::

    Saving and loading pre-trained instances with the ``bentoml.transformers`` APIs are supported starting release ``v1.0.17``.

Saving Pre-Trained Models and Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-trained models can be saved either as a pipeline or as a standalone model. Other pre-trained instances from Transformers,
such as tokenizers, preprocessors, and feature extractors, can also be saved as standalone models using the ``bentoml.transformers.save_model`` API.

.. code-block:: python
    :caption: `train.py`

    import bentoml
    from transformers import AutoTokenizer

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    bentoml.transformers.save_model("speecht5_tts_processor", processor)
    bentoml.transformers.save_model("speecht5_tts_model", model, signatures={"generate_speech": {"batchable": False}})
    bentoml.transformers.save_model("speecht5_tts_vocoder", vocoder)

To load the pre-trained instances for testing and debugging, use :code:`bentoml.transformers.load_model` with the same tags.

Serving Pretrained Models and Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-trained models and instances can be run either independently as Transformers framework runners or jointly in a custom runner. If you wish to
run them in isolated processes, use pre-trained models and instances as individual framework runners. On the other hand, if you wish to run them
in the same process, use pre-trained models and instances in a custom runner. Using a custom runner is typically more efficient as it can avoid
unnecessary overhead incurred during interprocess communication.

To use pre-trained models and instances as individual framework runners, simply get the models reference and convert them to runners using the
``to_runner`` method.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    import torch

    from bentoml.io import Text, NumpyNdarray
    from datasets import load_dataset

    proccessor_runner = bentoml.transformers.get("speecht5_tts_processor").to_runner()
    model_runner = bentoml.transformers.get("speecht5_tts_model").to_runner()
    vocoder_runner = bentoml.transformers.get("speecht5_tts_vocoder").to_runner()
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    svc = bentoml.Service("text2speech", runners=[proccessor_runner, model_runner, vocoder_runner])

    @svc.api(input=Text(), output=NumpyNdarray())
    def generate_speech(inp: str):
        inputs = proccessor_runner.run(text=inp, return_tensors="pt")
        speech = model_runner.generate_speech.run(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder_runner.run)
        return speech.numpy()

Alternatively, to use the pre-trained models and instances together in a custom runner, use the ``bentoml.transformers.get`` API to get the models
references and load them in a custom runner. The pretrained instances can then be used for inference in the custom runner.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    import torch

    from datasets import load_dataset


    processor_ref = bentoml.models.get("speecht5_tts_processor:latest")
    model_ref = bentoml.models.get("speecht5_tts_model:latest")
    vocoder_ref = bentoml.models.get("speecht5_tts_vocoder:latest")


    class SpeechT5Runnable(bentoml.Runnable):

        def __init__(self):
            self.processor = bentoml.transformers.load_model(processor_ref)
            self.model = bentoml.transformers.load_model(model_ref)
            self.vocoder = bentoml.transformers.load_model(vocoder_ref)
            self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        @bentoml.Runnable.method(batchable=False)
        def generate_speech(self, inp: str):
            inputs = self.processor(text=inp, return_tensors="pt")
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            return speech.numpy()


    text2speech_runner = bentoml.Runner(SpeechT5Runnable, name="speecht5_runner", models=[processor_ref, model_ref, vocoder_ref])
    svc = bentoml.Service("talk_gpt", runners=[text2speech_runner])

    @svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
    async def generate_speech(inp: str):
        return await text2speech_runner.generate_speech.async_run(inp)

Built-in Pipelines
------------------

Transformers pipelines are a high-level API for performing common natural language processing tasks using pre-trained transformer models.
See `Transformers Pipelines tutorial <https://huggingface.co/docs/transformers/pipeline_tutorial>`_ to learn more.

Saving a Pipeline
~~~~~~~~~~~~~~~~~

To save a Transformers Pipeline, first create a Pipeline object using the desired model and other pre-trained instances, and then save it to
the model store using the ``bentoml.transformers.save_model`` API. Transformers pipelines are callable objects, and thus the signatures of the
model are automatically saved as __call__ by default.

.. code-block:: python
    :caption: `train.py`

    import bentoml
    from transformers import pipeline

    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    bentoml.transformers.save_model(name="unmasker", pipeline=unmasker)

To load the pipeline for testing and debugging, use :code:`bentoml.transformers.load_model` with the :code:`unmasker:latest` tag.

Serving a Pipeline
~~~~~~~~~~~~~~~~~~

.. seealso::

   See :ref:`Building a Service <concepts/service:Service and APIs>` to learn more on creating a prediction service with BentoML.

To serve a Transformers pipeline, first get the pipeline reference using the ``bentoml.transformers.get`` API and convert it to a runner using
the ``to_runner`` method.

.. code-block:: python
    :caption: `service.py`

    import bentoml

    from bentoml.io import Text, JSON

    runner = bentoml.transformers.get("unmasker:latest").to_runner()

    svc = bentoml.Service("unmasker_service", runners=[runner])

    @svc.api(input=Text(), output=JSON())
    async def unmask(input_series: str) -> list:
        return await runner.async_run(input_series)

Custom Pipelines
----------------

Transformers custom pipelines allow users to define their own pre and post-process logic and customize how input data is forwarded to 
the model for inference.

.. seealso::

    `How to add a pipeline <https://huggingface.co/docs/transformers/main/en/add_new_pipeline>`_ from Hugging Face to learn more.

.. code-block:: python
    :caption: `train.py`
    
    from transformers import Pipeline

    class MyClassificationPipeline(Pipeline):
        def _sanitize_parameters(self, **kwargs):
            preprocess_kwargs = {}
            if "maybe_arg" in kwargs:
                preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
            return preprocess_kwargs, {}, {}

        def preprocess(self, text, maybe_arg=2):
            input_ids = self.tokenizer(text, return_tensors="pt")
            return input_ids

        def _forward(self, model_inputs):
            outputs = self.model(**model_inputs)
            return outputs

        def postprocess(self, model_outputs):
            return model_outputs["logits"].softmax(-1).numpy()

Saving a Custom Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

A custom pipeline first needs to be added to the Transformers supported tasks, :code:`SUPPORTED_TASKS` before it can be created with 
the Transformers :code:`pipeline` API.

.. code-block:: python
    :caption: `train.py`
    
    from transformers import pipeline
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    from transformers.pipelines import SUPPORTED_TASKS

    TASK_NAME = "my-classification-task"
    TASK_DEFINITION = {
        "impl": MyClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForSequenceClassification,),
        "default": {},
        "type": "text",
    }
    SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

    classifier = pipeline(
        task=TASK_NAME,
        model=AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
    )

Once a new pipeline is added to the Transformers supported tasks, it can be saved to the BentoML model store with the additional 
arguments of :code:`task_name` and :code:`task_definition`, the same arguments that were added to the Transformers :code:`SUPPORTED_TASKS` 
when creating the pipeline. :code:`task_name` and :code:`task_definition` will be saved as model options alongside the model.

.. code-block:: python
   :caption: `train.py`
    
    import bentoml

    bentoml.transformers.save_model(
        "my_classification_model",
        pipeline=classifier,
        task_name=TASK_NAME,
        task_definition=TASK_DEFINITION,
    )

Serving a Custom Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

To serve a custom pipeline, simply create a runner and service with the previously saved pipeline. :code:`task_name` and 
:code:`task_definition` will be automatically applied when initializing the runner.

.. code-block:: python
    :caption: `service.py`
    
    import bentoml

    from bentoml.io import Text, JSON

    runner = bentoml.transformers.get("my_classification_model:latest").to_runner()

    svc = bentoml.Service("my_classification_service", runners=[runner])

    @svc.api(input=Text(), output=JSON())
    async def classify(input_series: str) -> list:
        return await runner.async_run(input_series)

Adaptive Batching
-----------------

If the model supports batched interence, it is recommended to enable batching to take advantage of the adaptive batching capability 
in BentoML by overriding the :code:`signatures` argument with the method name (:code:`__call__`), :code:`batchable`, and :code:`batch_dim` 
configurations when saving the model to the model store . 

.. seealso::

   See :ref:`Adaptive Batching <guides/batching:Adaptive Batching>` to learn more.

.. code-block:: python
    :caption: `train.py`

    import bentoml

    bentoml.transformers.save_model(
        name="unmasker",
        pipeline=unmasker,
        signatures={
            "__call__": {
                "batchable": True,
                "batch_dim": 0,
            },
        },
    )

.. Serving on GPU
.. --------------

.. BentoML Transformers framework will enable inference on GPU if the hardware is available.

.. .. seealso::

..    See :ref:`Serving with GPU <guides/gpu:Serving with GPU>` to learn more.
