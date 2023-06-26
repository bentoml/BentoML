====================
Build with Rapid Iterations
====================

----------------
Start with BentoML
----------------

BentoML is a unified open source framework for building AI applications incorporating traditional ML, pre-trained AI models, Generative and Large Language Models.

Its strength lies in its extensive **integration** with a wide range of AI frameworks, thus guaranteeing support for your AI models regardless of their complexity. To learn about the various frameworks that integrate with BentoML, see the :doc:`Framework Guide </frameworks/index>`.

As BentoML is built with **Python**, it offers an intuitive coding experience for ML practitioners, seamlessly integrating with popular libraries within the Python ecosystem. This makes your coding experience both familiar and efficient.

BentoML embraces **open standards** for AI applications and promotes **best practices** to enhance the quality of your work. It offers a versatile framework capable of unifying **online**, **offline**, and **streaming** workloads.

In essence, BentoML aims to simplifies the process of building ML models, ensuring your team can focus on what truly matters: creating AI applications that solve real-world problem.

---------------------
Building Applications
---------------------

Let's delve into how to **BUILD** an NLP application that categorizes and summarizes text with HuggingFace’s Transformers. We will be demonstrating with `Transformers NLP Service <https://github.com/bentoml/transformers-nlp-service>`_ project.

Start by cloning the repository:

.. code-block:: bash

   git clone https://github.com/bentoml/transformers-nlp-service.git

Then, install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt

To serve your model as an HTTP server, utilize the ``bentoml serve`` CLI command. This starts a local server at `localhost:3000`, making your model accessible as a web service.

Next, build your Bento:

.. code-block:: bash

   bentoml build

Your Bento is now ready! A Bento, in BentoML, is the application artifact. It packages your program's source code, models, configs, and dependencies. This Bento can be distributed and deployed across a variety of platforms.

To learn how to deploy your Bento with BentoCloud, read :doc:`Getting Started -- Ship To Production <ship>`

.. _bento-gallery:

-------------
Bento Gallery
-------------

The `Bento Gallery <https://bentoml.com/gallery>`_ is a curated collection showcasing various types of ML models built and served using BentoML. Explore, learn, and draw inspiration from these showcased projects.

.. grid:: 2 3 3 3
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: OpenLLM
        :link: https://github.com/bentoml/OpenLLM
        :link-type: url

        An open platform for operating large language models (LLMs) in production.
        Fine-tune, serve, deploy, and monitor any LLMs with ease.

    .. grid-item-card:: CLIP
        :link: https://github.com/bentoml/CLIP-API-service
        :link-type: url

        Discover the effortless integration of OpenAI's innovative CLIP model with BentoML.

    .. grid-item-card:: Transformer
        :link: https://github.com/bentoml/transformers-nlp-service
        :link-type: url

        A modular, composable, and scalable solution for building NLP services with Transformers

    .. grid-item-card:: Pneumonia Detection
        :link: https://github.com/bentoml/Pneumonia-Detection-Demo
        :link-type: url

        Healthcare AI 🫁🔍 built with BentoML and fine-tuned Vision Transformer (ViT) model

    .. grid-item-card:: Fraud Detection
        :link: https://github.com/bentoml/Fraud-Detection-Model-Serving
        :link-type: url

        Online model serving with Fraud Detection model trained with XGBoost on IEEE-CIS dataset

    .. grid-item-card:: Optical Character Recognition (OCR)
        :link: https://github.com/bentoml/OCR-as-a-Service
        :link-type: url

        An efficient solution for converting PDFs into text 🚀
