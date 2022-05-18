========================
Huggingface Transformers
========================

Users can now use Transformers with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml

   # `import` a pretrained model and retrieve coresponding tag:
   tag = bentoml.transformers.import_from_huggingface_hub("distilbert-base-uncased-finetuned-sst-2-english")

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # Load a given model under `Runner` abstraction with `load_runner`
   runner = bentoml.transformers.load_runner(tag, tasks="text-classification")

   batched_sentence = [
      "I love you and I want to spend my whole life with you",
      "I hate you, Lyon, you broke my heart.",
   ]
   runner.run_batch(batched_sentence)

We also offer :code:`import_from_huggingface_hub` which enables users to import model from `HuggingFace Models <https://huggingface.co/models>`_ and use it with BentoML:

.. code-block:: python

   import bentoml
   import requests
   from PIL import Image

   tag = bentoml.transformers.import_from_huggingface_hub("google/vit-large-patch16-224")

   runner = bentoml.transformers.load_runner(
       tag,
       tasks="image-classification",
       device=-1,
       feature_extractor="google/vit-large-patch16-224",
       model_store=modelstore,
   )
   url = "http://images.cocodataset.org/val2017/000000039769.jpg"
   image = Image.open(requests.get(url, stream=True).raw)
   res = runner.run_batch(image)

.. note::

   You can find more examples for **Transformers** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.transformers

.. autofunction:: bentoml.transformers.save

.. autofunction:: bentoml.transformers.load

.. autofunction:: bentoml.transformers.load_runner

.. autofunction:: bentoml.transformers.import_from_huggingface_hub
