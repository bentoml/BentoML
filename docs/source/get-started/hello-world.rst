===========
Hello world
===========

.. meta::
    :description lang=en:
        Serve a simple text summarization model with BentoML.

This tutorial demonstrates how to serve a text summarization model from Hugging Face. You will do the following in this tutorial:

- Set up the BentoML environment
- Create a BentoML Service
- Serve the model locally

You can find the source code in the `quickstart <https://github.com/bentoml/quickstart>`_ GitHub repository.

Set up the environment
----------------------

1. Clone the project repository.

   .. code-block:: bash

      git clone https://github.com/bentoml/quickstart.git
      cd quickstart

2. Create a virtual environment and activate it.

   .. tab-set::

        .. tab-item:: Mac/Linux

            .. code-block:: bash

               python3 -m venv quickstart
               source quickstart/bin/activate

        .. tab-item:: Windows

            .. code-block:: bash

               python -m venv quickstart
               quickstart\Scripts\activate

   .. note::

      We recommend you create a virtual environment for dependency isolation. If you don't want to set up a local development environment, skip to the :doc:`BentoCloud deployment document <cloud-deployment>`.

3. Install the dependencies.

   .. code-block:: bash

      # Recommend Python 3.11
      pip install -r requirements.txt

Create a BentoML Service
------------------------

You can define the serving logic of the model in a ``service.py`` file. Here is the example in this project:

.. code-block:: python
    :caption: `service.py`

    from __future__ import annotations
    import bentoml

    with bentoml.importing():
        from transformers import pipeline


    EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century."


    @bentoml.service
    class Summarization:
        def __init__(self) -> None:
            self.pipeline = pipeline('summarization')

        @bentoml.api
        def summarize(self, text: str = EXAMPLE_INPUT) -> str:
            result = self.pipeline(text)
            return f"Hello world! Here's your summary: {result[0]['summary_text']}"

In the ``Summarization`` class, the BentoML Service retrieves a pre-trained model and initializes a pipeline for text summarization. The ``summarize`` method serves as the API endpoint. It accepts a string input with a sample provided, processes it through the pipeline, and returns the summarized text.

In BentoML, a :doc:`Service </build-with-bentoml/services>` is a deployable and scalable unit, defined as a Python class using the ``@bentoml.service`` decorator. It can manage states and their lifecycle, and expose one or multiple APIs accessible through HTTP. Each API within the Service is defined using the ``@bentoml.api`` decorator, specifying it as a Python function.

The ``bentoml.importing()`` context manager is used to handle import statements for dependencies required during serving but may not be available in other situations.

Serve the model locally
-----------------------

1. Run ``bentoml serve service:<service_class_name>`` to start the BentoML server.

   .. code-block:: bash

      $ bentoml serve service:Summarization

      2024-02-02T07:16:14+0000 [WARNING] [cli] Converting 'Summarization' to lowercase: 'summarization'.
      2024-02-02T07:16:15+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:Summarization" listening on http://localhost:3000 (Press CTRL+C to quit)

2. You can call the exposed ``summarize`` endpoint at http://localhost:3000.

   .. tab-set::

       .. tab-item:: CURL

           .. code-block:: bash

               curl -X 'POST' \
                   'http://localhost:3000/summarize' \
                   -H 'accept: text/plain' \
                   -H 'Content-Type: application/json' \
                   -d '{
                   "text": "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson'\''s cat, Whiskers, performed what witnesses are calling a '\''miraculous and gravity-defying leap.'\'' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson'\''s backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as '\''The Leap of the Century."
               }'

       .. tab-item:: Python client

           .. code-block:: python

               import bentoml

               with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                   result = client.summarize(
                       text="Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century.'"
                   )
                   print(result)

       .. tab-item:: Swagger UI

           Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

           .. image:: ../_static/img/get-started/hello-world/service-ui.png
              :alt: BentoML hello world example Swagger UI

   Expected output:

   .. code-block:: bash

       Hello world! Here's your summary: Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly . The event is now being investigated by scientists for potential breaches in the laws of physics . Local authorities considering a town festival to celebrate what is being hailed as 'The Leap of the Century'

What's next
-----------

- :doc:`Batch requests <adaptive-batching>`
- :doc:`Load your own model </build-with-bentoml/model-loading-and-management>`
- :doc:`Create a Docker image <packaging-for-deployment>`
- :doc:`Deploy to the cloud <cloud-deployment>`
