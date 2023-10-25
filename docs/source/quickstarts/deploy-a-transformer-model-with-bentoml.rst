=======================================
Deploy a Transformer model with BentoML
=======================================

This quickstart demonstrates how to build a text summarization application with a Transformer model (`sshleifer/distilbart-cnn-12-6 <https://huggingface.co/sshleifer/distilbart-cnn-12-6>`_)
from the Hugging Face Model Hub. It helps you become familiar with the core concepts of BentoML and gain a basic understanding of the model serving
lifecycle in BentoML.

Specifically, you will do the following in this tutorial:

- Set up the BentoML environment
- Download a model
- Create a BentoML Service
- Build a Bento
- Serve the Bento
- (Optional) Containerize the Bento with Docker

.. note::

   All the project files are stored on the `quickstart <https://github.com/bentoml/quickstart>`_ GitHub repository.

Prerequisites
-------------

- Make sure you have Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- (Optional) Install `Docker <https://docs.docker.com/get-docker/>`_ if you want to containerize the Bento.
- (Optional) We recommend you create a virtual environment for dependency isolation for this quickstart. For more information about virtual environments in Python, see `Creation of virtual environments <https://docs.python.org/3/library/venv.html>`_.

Install dependencies
--------------------

Run the following command to install the required dependencies, which include BentoML, Transformers, and Pytorch (or TensorFlow 2.0).

.. code-block:: bash

   pip install bentoml transformers torch

Download a model to the local Model Store
-----------------------------------------

To create this text summarization AI application, you need to download the model first. This is done via a ``download_model.py`` script as below,
which uses the ``bentoml.transformers.save_model()`` function to import the model to the local Model Store. The BentoML Model Store is used for
managing all your local models as well as accessing them for serving.

.. code-block:: python
   :caption: `download_model.py`

   import transformers
   import bentoml

   model= "sshleifer/distilbart-cnn-12-6"
   task = "summarization"

   bentoml.transformers.save_model(
       task,
       transformers.pipeline(task, model=model),
       metadata=dict(model_name=model),
   )

Create and run this script to download the model.

.. code-block:: bash

   python download_model.py

.. note::

   It is possible to use pre-trained models directly with BentoML or import existing trained model files to BentoML.
   See :doc:`/concepts/model` to learn more.

The model should appear in the Model Store with the name ``summarization`` if the download is successful. You can retrieve this model later to
create a BentoML Service. Run ``bentoml models list`` to view all available models in the Model Store.

.. code-block:: bash

   $ bentoml models list

   Tag                                    Module                Size       Creation Time
   summarization:5kiyqyq62w6pqnry         bentoml.transformers  1.14 GiB   2023-07-10 11:57:40

.. note::

   All models downloaded to the Model Store are saved in the directory ``/home/user/bentoml/models/``. You can manage saved models via
   the ``bentoml models`` CLI command or Python API. For more information, see :ref:`concepts/model:Manage models`.

Create a BentoML Service
------------------------

With a ready-to-use model, you define a BentoML Service by creating a ``service.py`` file as below. This is where the serving logic is defined.

.. code-block:: python
   :caption: `service.py`

   import bentoml

   summarizer_runner = bentoml.models.get("summarization:latest").to_runner()

   svc = bentoml.Service(
       name="summarization", runners=[summarizer_runner]
   )

   @svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
   async def summarize(text: str) -> str:
       generated = await summarizer_runner.async_run(text, max_length=3000)
       return generated[0]["summary_text"]

This script creates a ``summarizer_runner`` instance from the previously downloaded model, retrieved through the ``bentoml.models.get()`` function.
A Runner in BentoML is a computational unit that encapsulates a machine learning model. It's designed for remote execution and independent scaling.
For more information, see :doc:`/concepts/runner`.

``bentoml.Service()`` wraps the Runner and creates a Service. A BentoML Service encapsulates various components including Runners and an API server.
It serves as the interface to the outside world, processing incoming requests and outgoing responses. A single Service can house multiple Runners,
enabling the construction of more complex machine learning applications. The diagram below provides a high-level representation of a BentoML Service:

.. image:: ../../_static/img/quickstarts/deploy-a-transformer-model-with-bentoml/service.png

The ``summarize()`` function, decorated with ``@svc.api()``, specifies the API endpoint for the Service and the logic to process the inputs and outputs.
For more information, see :doc:`/reference/api_io_descriptors`.

Run ``bentoml serve`` in your project directory to start the BentoML server.

.. code-block:: bash

   $ bentoml serve service:svc

   2023-07-10T12:13:33+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:svc" can be accessed at http://localhost:3000/metrics.
   2023-07-10T12:13:34+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:svc" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)

The server is now active at `http://0.0.0.0:3000 <http://0.0.0.0:3000/>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

         curl -X 'POST' \
            'http://0.0.0.0:3000/summarize' \
            -H 'accept: text/plain' \
            -H 'Content-Type: text/plain' \
            -d '$PROMPT' # Replace $PROMPT here with your prompt.

    .. tab-item:: Python

        .. code-block:: bash

         import requests

         response = requests.post(
            "http://0.0.0.0:3000/summarize",
            headers={
               "accept": "text/plain",
               "Content-Type": "text/plain",
            },
            data="$PROMPT", # Replace $PROMPT here with your prompt.
         )

         print(response.text)

    .. tab-item:: Browser

        Visit `http://0.0.0.0:3000 <http://0.0.0.0:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

        .. image:: ../../_static/img/quickstarts/deploy-a-transformer-model-with-bentoml/service-ui.png

See the following example that summarizes the concept of large language models.

Input:

.. code-block::

   A large language model (LLM) is a computerized language model, embodied by an artificial neural network using an enormous amount of "parameters" (i.e. "neurons" in its layers with up to tens of millions to billions "weights" between them), that are (pre-)trained on many GPUs in relatively short time due to massive parallel processing of vast amounts of unlabeled texts containing up to trillions of tokens (i.e. parts of words) provided by corpora such as Wikipedia Corpus and Common Crawl, using self-supervised learning or semi-supervised learning, resulting in a tokenized vocabulary with a probability distribution. LLMs can be upgraded by using additional GPUs to (pre-)train the model with even more parameters on even vaster amounts of unlabeled texts.

Output by the text summarization model:

.. code-block::

   A large language model (LLM) is a computerized language model, embodied by an artificial neural network using an enormous amount of "parameters" in its layers with up to tens of millions to billions "weights" between them . LLMs can be upgraded by using additional GPUs to (pre-)train the model with even more parameters on even vaster amounts of unlabeled texts .

Build a Bento
-------------

Once the model is functioning properly, you can package it into the standard distribution format in BentoML, also known as a "Bento".
It is a self-contained archive that contains all the source code, model files, and dependencies required to run the Service.

To build a Bento, you need a configuration YAML file (by convention, it’s ``bentofile.yaml``). This file defines the build options, such as dependencies,
Docker image settings, and models. The example file below only lists the basic information required to build a Bento,
including the Service, Python files, dependencies, and model. See :ref:`Bento build options <concepts/bento:Bento build options>` to learn more.

.. code-block:: yaml
   :caption: `bentofile.yaml`

   service: 'service:svc'
   include:
     - '*.py'
   python:
     packages:
       - torch
       - transformers
   models:
     - summarization:latest

Run ``bentoml build`` in your project directory (which should contain ``download_model.py``, ``service.py``, and ``bentofile.yaml`` now) to build the Bento. You can find all created Bentos in ``/home/user/bentoml/bentos/``.

.. code-block:: bash

   $ bentoml build

   Building BentoML service "summarization:ulnyfbq66gagsnry" from build context "/Users/demo/Documents/bentoml-demo".
   Packing model "summarization:5kiyqyq62w6pqnry"

   ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
   ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
   ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
   ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
   ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
   ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

   Successfully built Bento(tag="summarization:ulnyfbq66gagsnry").

   Possible next steps:

    * Containerize your Bento with `bentoml containerize`:
       $ bentoml containerize summarization:ulnyfbq66gagsnry

    * Push to BentoCloud with `bentoml push`:
       $ bentoml push summarization:ulnyfbq66gagsnry

View all available Bentos:

.. code-block:: bash

   $ bentoml list

   Tag                               Size       Creation Time
   summarization:ulnyfbq66gagsnry    1.25 GiB   2023-07-10 15:28:51

.. note::

   Bentos are the deployment unit in BentoML, one of the most important artifacts to keep track of in your model deployment workflow.
   BentoML provides CLI commands and APIs for managing Bentos. See :ref:`Managing Bentos <concepts/bento:Manage Bentos>` to learn more.

Serve and deploy the Bento
--------------------------

Once the Bento is ready, you can use ``bentoml serve`` to serve it as an HTTP server in production. Note that if you have multiple versions of the same model, you can change the ``latest`` tag to the corresponding version.

.. code-block:: bash

   $ bentoml serve summarization:latest

   2023-07-10T15:36:58+0800 [INFO] [cli] Environ for worker 0: set CPU thread count to 12
   2023-07-10T15:36:58+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "summarization:latest" can be accessed at http://localhost:3000/metrics.
   2023-07-10T15:36:59+0800 [INFO] [cli] Starting production HTTP BentoServer from "summarization:latest" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)

You can containerize the Bento with Docker. When creating the Bento, a Dockerfile is created automatically at ``/home/user/bentoml/bentos/<bento_name>/<tag>/env/docker/``. To create a Docker image based on this example model, simply run:

.. code-block:: bash

   bentoml containerize summarization:latest

.. note::

   For Mac computers with Apple silicon, you can specify the ``--platform`` option to avoid potential compatibility issues with some Python libraries.

   .. code-block:: bash

      bentoml containerize --platform=linux/amd64 summarization:latest

The Docker image’s tag is the same as the Bento tag by default. View the created Docker image:

.. code-block:: bash

   $ docker images

   REPOSITORY                    TAG                IMAGE ID       CREATED         SIZE
   summarization                 ulnyfbq66gagsnry   da287141ef3e   7 seconds ago   2.43GB

Run the Docker image locally:

.. code-block:: bash

   docker run -it --rm -p 3000:3000 summarization:ulnyfbq66gagsnry serve

With the Docker image, you can run the model on Kubernetes and create a Kubernetes Service to expose it so that your users can interact with it.

If you prefer a serverless platform to build and operate AI applications, you can deploy Bentos to BentoCloud. It gives AI application developers a collaborative environment
and a user-friendly toolkit to ship and iterate AI products. For more information, see :doc:`/bentocloud/how-tos/deploy-bentos`.

.. note::

   BentoML provides a GitHub Action to help you automate the process of building Bentos and deploying them to the cloud. For more information, see :doc:`/guides/github-actions`.

See also
--------

- :doc:`/quickstarts/install-bentoml`
- :doc:`/quickstarts/deploy-a-large-language-model-with-openllm-and-bentoml`
