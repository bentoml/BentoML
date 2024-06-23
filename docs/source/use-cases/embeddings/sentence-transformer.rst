====================
Sentence Transformer
====================

In natural language processing (NLP), embeddings enable computers to understand the underlying semantics of language by transforming words, phrases, or even documents into numerical vectors. It covers a variety of use cases, from recommending products based on textual descriptions to translating languages and identifying relevant images through semantic understanding.

This document demonstrates how to build a sentence embedding application Sentence Transformer using BentoML. It uses the `all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_ model, a specific kind of language model developed for generating embeddings. Due to its smaller size, all-MiniLM-L6-v2 is efficient in terms of computational resources and speed, making it an ideal choice for embedding generation in environments with limited resources.

All the source code in this tutorial is available in the `BentoSentenceTransformers GitHub repository <https://github.com/bentoml/BentoSentenceTransformers>`_.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

Clone the project repository and install all the dependencies.

.. code-block:: bash

    git clone https://github.com/bentoml/BentoSentenceTransformers.git
    cd BentoSentenceTransformers
    pip install -r requirements.txt

Create a BentoML Service
------------------------

Define a :doc:`BentoML Service </guides/services>` to use a model for generating sentence embeddings. The example ``service.py`` file in this project uses ``sentence-transformers/all-MiniLM-L6-v2``:

.. code-block:: python
    :caption: `service.py`

    from __future__ import annotations

    import typing as t

    import numpy as np
    import bentoml


    SAMPLE_SENTENCES = [
        "The sun dips below the horizon, painting the sky orange.",
        "A gentle breeze whispers through the autumn leaves.",
        "The moon casts a silver glow on the tranquil lake.",
        "A solitary lighthouse stands guard on the rocky shore.",
        "The city awakens as morning light filters through the streets.",
        "Stars twinkle in the velvety blanket of the night sky.",
        "The aroma of fresh coffee fills the cozy kitchen.",
        "A curious kitten pounces on a fluttering butterfly."
    ]

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    @bentoml.service(
        traffic={
            "timeout": 60
            "concurrency": 32,
        },
        resources={
            "gpu": "1",
            "gpu_type": "nvidia-tesla-t4",
        },
    )
    class SentenceTransformers:

        def __init__(self) -> None:

            import torch
            from sentence_transformers import SentenceTransformer, models

            # Load model and tokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # define layers
            first_layer = SentenceTransformer(MODEL_ID)
            pooling_model = models.Pooling(first_layer.get_sentence_embedding_dimension())
            self.model = SentenceTransformer(modules=[first_layer, pooling_model])
            print("Model loaded", "device:", self.device)


        @bentoml.api(batchable=True)
        def encode(
            self,
            sentences: t.List[str] = SAMPLE_SENTENCES,
        ) -> np.ndarray:
            print("encoding sentences:", len(sentences))
            # Tokenize sentences
            sentence_embeddings= self.model.encode(sentences)
            return sentence_embeddings

Here is a breakdown of the Service code:

- The script uses the ``@bentoml.service`` decorator to annotate the ``SentenceTransformers`` class as a BentoML Service with timeout and memory specified. You can set more configurations as needed.
- ``__init__`` loads the model and tokenizer when an instance of the ``SentenceTransformers`` class is created. The model is loaded onto the appropriate device (GPU if available, otherwise CPU).
- The model consists of two layers: The first layer is the pre-trained MiniLM model (``all-MiniLM-L6-v2``), and the second layer is a pooling layer to aggregate word embeddings into sentence embeddings.
- The ``encode`` method is defined as a BentoML API endpoint. It takes a list of sentences as input and uses the sentence transformer model to generate sentence embeddings. The returned embeddings are NumPy arrays.

Run ``bentoml serve`` in your project directory to start the Service.

.. code-block:: bash

    $ bentoml serve service:SentenceTransformers

    2023-12-27T07:49:25+0000 [WARNING] [cli] Converting 'all-MiniLM-L6-v2' to lowercase: 'all-minilm-l6-v2'.
    2023-12-27T07:49:26+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:SentenceTransformers" listening on http://localhost:3000 (Press CTRL+C to quit)
    Model loaded device: cuda

The server is active at `http://localhost:3000 <http://localhost:3000>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/encode' \
                -H 'accept: application/json' \
                -H 'Content-Type: application/json' \
                -d '{
                "sentences": [
                    "hello world"
                ]
            }'

    .. tab-item:: Python client

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                result = client.encode(
                    sentences=[
                            "hello world"
                    ],
                )

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

        .. image:: ../../_static/img/use-cases/embeddings/sentence-embeddings/service-ui.png

Expected output:

.. code-block:: bash

    [
      [
        -0.19744610786437988,
        0.17766520380973816,
        ......
        0.2229892462491989,
        0.17298966646194458
      ]
    ]

Deploy to BentoCloud
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits.

First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for your application. It is used for packaging your application into a Bento. Here is an example file in the project:

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:SentenceTransformers"
    labels:
      owner: bentoml-team
      project: gallery
    include:
    - "*.py"
    python:
      requirements_txt: "./requirements.txt"
    docker:
      env:
        NORMALIZE : "True"

:ref:`Create an API token with Developer Operations Access to log in to BentoCloud <bentocloud/how-tos/manage-access-token:create an api token>`, then run the following command to deploy the project.

.. code-block:: bash

    bentoml deploy .

Once the Deployment is up and running on BentoCloud, you can access it via the exposed URL.

.. image:: ../../_static/img/use-cases/embeddings/sentence-embeddings/sentence-embedding-bentocloud.png

.. note::

   For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image</guides/containerization>`.
