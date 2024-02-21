===========
Get started
===========

BentoCloud offers serverless infrastructure tailored for AI inference, allowing you to efficiently deploy, manage, and scale any machine learning (ML) models in the cloud. It operates in conjunction with BentoML, an open-source model serving framework, to facilitate the easy creation and deployment of high-performance AI API services with custom code. As the original creators of BentoML and its ecosystem tools like OpenLLM, we seek to improve cost efficiency of your inference workload with our
serverless infrastructure optimized for GPUs and fast autoscaling.

Specifically, BentoCloud features:

- Optimized infrastructure for deploying any model, including the latest large language models (LLMs), Stable Diffusion models, and user-customized models built with various ML frameworks.
- Autoscaling with scale-to-zero support so you only pay for what you use.
- Flexible APIs for continuous integration and deployments (CI/CD).
- Built-in observability tools for monitoring model performance and troubleshooting.

Access BentoCloud
-----------------

To gain access to BentoCloud, visit the `BentoML website <https://www.bentoml.com/>`_ to sign up.

Once you have your BentoCloud account, do the following to get started:

1. Install BentoML by running ``pip install bentoml``. See :doc:`/get-started/installation` for details.
2. Create an :doc:`API token with Developer Operations Access </bentocloud/how-tos/manage-access-token>`.
3. Log in to BentoCloud with the ``bentoml cloud login`` command, which will be displayed on the BentoCloud console after you create the API token.

Deploy your first model
-----------------------

Perform the following steps to quickly deploy an example application on BentoCloud. It is a summarization service powered by a Transformer model `sshleifer/distilbart-cnn-12-6 <https://huggingface.co/sshleifer/distilbart-cnn-12-6>`_.

1. Install the dependencies.

   .. code-block:: bash

      pip install bentoml torch transformers

2. Create a BentoML Service in a ``service.py`` file as below. The pre-trained model is pulled from Hugging Face.

   .. code-block:: python

      from __future__ import annotations
      import bentoml
      from transformers import pipeline


      EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small \
      town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
      Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
      Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
      a record-breaking 20 feet into the air to catch a fly. The event, which took \
      place in Thompson's backyard, is now being investigated by scientists for potential \
      breaches in the laws of physics. Local authorities are considering a town festival \
      to celebrate what is being hailed as 'The Leap of the Century."


      @bentoml.service(
          resources={"cpu": "2"},
          traffic={"timeout": 10},
      )
      class Summarization:
          def __init__(self) -> None:
              self.pipeline = pipeline('summarization')

          @bentoml.api
          def summarize(self, text: str = EXAMPLE_INPUT) -> str:
              result = self.pipeline(text)
              return result[0]['summary_text']

   .. note::

      You can test this Service locally by running ``bentoml serve service:Summarization``. For details of the Service, see :doc:`/get-started/quickstart`.

3. Create a ``bentofile.yaml`` file as below.

   .. code-block:: yaml

        service: 'service:Summarization'
        labels:
          owner: bentoml-team
          project: gallery
        include:
        - '*.py'
        python:
          packages:
            - torch
            - transformers

4. Deploy the application to BentoCloud. The deployment status is displayed both in your terminal and the BentoCloud console.

   .. code-block:: bash

      bentoml deploy .

5. On the BentoCloud console, navigate to the **Deployments** page, and click your Deployment. Once it is up and running, interact with it using the Form, BentoML Python client, or CURL command on the **Playground** tab. The sample input already provides a new article and you can summarize it with the application.

   .. image:: ../_static/img/bentocloud/get-started/bentocloud-playground-quickstart.png

Resources
---------

If you are a first-time user of BentoCloud, we recommend you read the following documents to get familiar with BentoCloud:

- Deploy :doc:`example projects </use-cases/index>` to BentoCloud
- :doc:`/bentocloud/how-tos/manage-deployments`
