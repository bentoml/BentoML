============
Hugging Face
============

`🤗 Transformers <https://huggingface.co/docs/transformers/main/en/index>`_ is a library that helps download and fine-tune popular 
pretrained models for common machine learning tasks. BentoML provides native support for serving and deploying models trained from 
Transformers.

Compatibility 
-------------

BentoML requires Transformers version 4 or above. For other versions of Transformers, consider using a 
:ref:`concepts/runner:Custom Runner`.

When constructing a :ref:`bentofile.yaml <concepts/bento:Bento Build Options>`, include `transformers` and the machine learning 
framework of the model, e.g. `pytorch`, `tensorflow`, or `jax`.

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


Fined-tuned Models
------------------

Fine-tuning pretrained models is a powerful practice that allows users to save computation cost and adapt state-of-the-art models to their 
domain specific dataset. Transformers offers a variety of libraries for fine-tuning pretrained models. The example below fine-tunes a BERT 
model with Yelp review dataset. To learn more, refer to the Transformers guide on 
`fine-tuning pretrained models <https://huggingface.co/docs/transformers/main/en/training>`_.

.. code-block:: python
    :caption: `train.py`

    from datasets import load_dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForMaskedLM.from_pretrained("bert-base-cased", num_labels=5)

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
    )

    trainer.train()

Saving a Fine-tuned Model
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the model is fine-tuned, create a Transformers 
`Pipeline <https://huggingface.co/docs/transformers/main/en/pipeline_tutorial>`_ with the model and save to the BentoML model 
store. By design, only Pipelines can be saved with the BentoML Transformers framework APIs. Models, tokenizers, feature extractors, 
and processors, need to be a part of the pipeline first before they can be saved. Transformers pipelines are callable objects therefore 
the signatures of the model are saved as :code:`__call__` by default.

.. code-block:: python
    :caption: `train.py`

    import bentoml
    from transformers import pipeline

    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    bentoml.transformers.save_model(name="unmasker", pipeline=unmasker)

To load the model for testing and debugging, use :code:`bentoml.transformers.load_model` with the :code:`unmasker:latest` tag.

Serving a Fined-tuned Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a BentoML service with the previously saved `unmasker` pipeline using the Transformers framework APIs.

.. seealso::

   See :ref:`Building a Service <concepts/service:Service and APIs>` to learn more on creating a prediction service with BentoML.

.. code-block:: python
    :caption: `service.py`

    import bentoml

    from bentoml.io import Text, JSON

    runner = bentoml.transformers.get("unmasker:latest").to_runner()

    svc = bentoml.Service("unmasker_service", runners=[runner])

    @svc.api(input=Text(), output=JSON())
    def unmask(input_series: str) -> list:
        return runner.run(input_series)

Pretrained Models
-----------------

Using pretrained models from the Hugging Face does not require saving the model first in the BentoML model store. A custom runner 
can be implemented to download and run pretrained models at runtime.

.. seealso::

   See :ref:`Custom Runner <concepts/runner:Custom Runner>` to learn more.

Serving a Pretrained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    :caption: `service.py`

    import bentoml

    from bentoml.io import Text, JSON
    from transformers import pipeline

    class PretrainedModelRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            self.unmasker = pipeline(task="fill-mask", model="distilbert-base-uncased")

        @bentoml.Runnable.method(batchable=False)
        def __call__(self, input_text):
            return self.unmasker(input_text)

    runner = bentoml.Runner(PretrainedModelRunnable, name="pretrained_unmasker")

    svc = bentoml.Service('pretrained_unmasker_service', runners=[runner])

    @svc.api(input=Text(), output=JSON())
    def unmask(input_series: str) -> list:
        return runner.run(input_series)

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
    def classify(input_series: str) -> list:
        return runner.run(input_series)

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
