==================
Build with BentoML
==================

BentoCloud represents an important component in the BentoML ecosystem. To build your application and deploy it on BentoCloud,
the first step is to convert your model into a standardized distribution format using BentoML.

Understand BentoML
------------------

BentoML is a unified open-source framework for building reliable, scalable, and cost-efficient AI applications. It encompasses everything from model
serving to application packaging and production deployment.

AI application developers are the main audience for BentoML. As BentoML is built with **Python**, it offers an intuitive coding experience for machine learning (ML) practitioners,
integrating seamlessly with popular libraries within the Python ecosystem. Whether you are building or integrating AI applications through various frameworks like PyTorch,
TensorFlow, and Keras, BentoML caters to a wide range of development needs.

With BentoML, developers can get native support for traditional ML models, pre-trained AI models, generative AI models, and large language models (LLMs).
It integrates smoothly with popular tools like MLFlow, Kubeflow, and Triton. This extensive **integration** with popular frameworks and tools helps complete the production
AI stack, facilitating a smooth development process. For more information, see the :doc:`Framework Guide </frameworks/index>`.

BentoML embraces **open standards** for AI applications and promotes **best practices** to enhance the quality of your work. Its versatile framework is capable of unifying **online**,
**offline**, and **streaming** workloads, offering a multifaceted approach to AI application development and deployment.

In essence, BentoML aims to simplify the process of building ML models, allowing your team to focus on what truly matters: creating AI applications that solve real-world problems.
By providing an integrated and user-friendly framework, BentoML stands as an essential tool for anyone looking to develop state-of-the-art AI solutions.

For more information, see :doc:`What is BentoML </overview/what-is-bentoml>`.

Build an AI application
-----------------------

Perform the following steps to build a simple text summarization application with a Transformer model from the Hugging Face Model Hub.
All the project files are stored on the `quickstart <https://github.com/bentoml/quickstart>`_ GitHub repository.

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/bentoml/quickstart.git

2. Install the required dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Download the Transformer model.

   .. code-block:: bash

      python download_model.py

4. Serve your model as an HTTP server. This starts a local server at `http://0.0.0.0:3000 <http://0.0.0.0:3000/>`_, making your model accessible as a web service.
   
   .. code-block:: bash

      bentoml serve service:svc

5. Build your Bento. In the BentoML framework, a Bento is a deployable artifact that contains your application's source code, models, configurations, and dependencies.

   .. code-block:: bash

      bentoml build

After your Bento is ready, you can push your Bento to BentoCloud or containerize it with Docker and deploy it on a variety of platforms.
For more information, see this :doc:`quickstart in the BentoML documentation </quickstarts/deploy-a-transformer-model-with-bentoml>`.

To learn how to deploy your Bento with BentoCloud, read :doc:`ship`.

.. _bento-gallery:

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
