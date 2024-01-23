===================
Sentence embeddings
===================

In natural language processing (NLP), embeddings enable computers to understand the underlying semantics of language by transforming words, phrases, or even documents into numerical vectors. It covers a variety of use cases, from recommending products based on textual descriptions to translating languages and identifying relevant images through semantic understanding.

This document demonstrates how to build a sentence embedding application using BentoML. It uses the `all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_ model, a specific kind of language model developed for generating embeddings. Due to its smaller size, all-MiniLM-L6-v2 is efficient in terms of computational resources and speed, making it an ideal choice for embedding generation in environments with limited resources.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

.. code-block:: bash

    pip install torch transformers "bentoml>=1.2.0a0"

Create a BentoML Service
------------------------

Define a :doc:`BentoML Service </guides/services>` to use a model (``sentence-transformers/all-MiniLM-L6-v2`` in this example) for generating sentence embeddings. You can do this by creating a ``service.py`` file as below.

.. code-block:: python
    :caption: `service.py`

    import numpy as np
    import torch
    import bentoml
    from sentence_transformers import SentenceTransformer, models

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    @bentoml.service(
        traffic={"timeout": 60},
        resources={"memory": "2Gi"},
    )
    class SentenceEmbedding:

        def __init__(self) -> None:

            # Load model and tokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Define layers
            first_layer = SentenceTransformer(MODEL_ID)
            pooling_model = models.Pooling(first_layer.get_sentence_embedding_dimension())
            self.model = SentenceTransformer(modules=[first_layer, pooling_model])
            print("Model loaded", "device:", self.device)


        @bentoml.api(batchable=True)
        def encode(
            self,
            sentences: list[str],
        ) -> np.ndarray:
            # Tokenize sentences
            sentence_embeddings= self.model.encode(sentences)
            return sentence_embeddings

    if __name__ == "__main__":
        SentenceEmbedding.serve_http()

Here is a breakdown of the Service code:

- The script uses the ``@bentoml.service`` decorator to annotate the ``SentenceEmbedding`` class as a BentoML Service with timeout and memory specified. You can set more configurations as needed.
- ``__init__`` loads the model and tokenizer when an instance of the ``SentenceEmbedding`` class is created. The model is loaded onto the appropriate device (GPU if available, otherwise CPU).
- The model consists of two layers: The first layer is the pre-trained MiniLM model (``all-MiniLM-L6-v2``), and the second layer is a pooling layer to aggregate word embeddings into sentence embeddings.
- The ``encode`` method is defined as a BentoML API endpoint. It takes a list of sentences as input and uses the sentence transformer model to generate sentence embeddings. The returned embeddings are NumPy arrays.

Run ``bentoml serve`` in your project directory to start the Service.

.. code-block:: bash

    $ bentoml serve service:SentenceEmbedding

    2023-12-27T07:49:25+0000 [WARNING] [cli] Converting 'all-MiniLM-L6-v2' to lowercase: 'all-minilm-l6-v2'.
    2023-12-27T07:49:26+0000 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:SentenceEmbedding" can be accessed at http://localhost:3000/metrics.
    2023-12-27T07:49:26+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:SentenceEmbedding" listening on http://localhost:3000 (Press CTRL+C to quit)
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

    .. tab-item:: BentoML client

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

Deploy to production
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability.

First, specify a configuration YAML file (``bentofile.yaml``) as below to define the build options for your application. It is used for packaging your application into a Bento.

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:SentenceEmbedding"
    labels:
      owner: bentoml-team
      project: gallery
    include:
    - "*.py"
    python:
      packages:
        - torch
        - transformers

Make sure you :doc:`have logged in to BentoCloud </bentocloud/how-tos/manage-access-token>`, then run the following command in your project directory to deploy the application to BentoCloud. Under the hood, this commands automatically builds a Bento, push the Bento, and deploy it on BentoCloud.

.. code-block:: bash

    bentoml deploy .

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

.. note::

   Alternatively, you can use BentoML to generated an :doc:`OCI-compliant image for a more custom deployment </guides/containerization>`.
