=========
Ecosystem
=========

The BentoML team works to offer a suite of tools and platforms to
streamline the deployment and management of AI applications in
production. This document lists some of the highlighted ones in the
BentoML ecosystem.

BentoCloud
----------

BentoCloud is a fully-managed platform for building and operating AI
applications. It's designed to accelerate AI application development by
providing efficient workflows to deploy and scale a wide range of
machine learning models.

As a serverless solution, BentoCloud allows teams to run their AI
applications on fully-managed infrastructure that they can customize
based on their budget needs. It supports autoscaling (including
scale-to-zero) and provides great observability for monitoring. These
enable developers to focus on building AI applications without dealing
with the intricacies of the underlying infrastructure.

To use BentoCloud, BentoML users can package their models into Bentos
and push them to BentoCloud for better management and deployment. For
more information, see :doc:`the BentoCloud documentation </bentocloud/getting-started/understand-bentocloud>`.

OpenLLM
-------

OpenLLM is an open-source platform designed to simplify the deployment
and fine-tuning of large language models (LLMs). It allows you to run
inference with any open-source LLMs, deploy models to the cloud or
on-premises, and build powerful AI applications.

OpenLLM is aimed at empowering software engineers to better fine-tune,
serve, and deploy their models to production. It supports a wide range
of open-source LLMs, such as StableLM, Falcon, Dolly, Flan-T5, ChatGLM,
StarCoder, OPT, and Llama. It integrates seamlessly with BentoML,
LangChain, and Hugging Face, allowing for the composition or chaining of
LLM inferences with different AI models.

OpenLLM provides first-class support for BentoML. To use OpenLLM,
BentoML users can easily integrate OpenLLM models as BentoML Runners in
their Services. For more information, see the `OpenLLM GitHub
repository <https://github.com/bentoml/OpenLLM>`_.

OneDiffusion
------------

OneDiffusion is an open-source, all-in-one platform specially designed
to streamline the deployment of diffusion models. It supports both pretrained
and fine-tuned diffusion models with LoRA adapters, allowing you to run a variety of
image generation tasks with ease and flexibility. As it is integrated seamlessly
with the BentoML framework, you can use OneDiffusion to deploy any diffusion model to
the cloud or on-premises, and build powerful and scalable AI applications.

OneDiffusion currently supports Stable Diffusion and Stable Diffusion XL models.
More models (for example, ControlNet and DeepFloyd IF) will be supported in the future.
For more information, see the `OneDiffusion GitHub repository <https://github.com/bentoml/OneDiffusion>`_.

Yatai
-----

Yatai is the Kubernetes deployment operator for BentoML, allowing DevOps teams to seamlessly
integrate BentoML into their GitOps practices. It simplifies the deployment and scaling of ML services,
offering a subset of scalability features from BentoCloud. Designed with cloud-native principles,
Yatai leverages Custom Resource Definitions (CRDs) like `BentoDeployment <https://docs.yatai.io/en/latest/concepts/bentodeployment_crd.html>`_
to fit ML services into existing workflows. For more information, see the `Yatai
documentation <https://docs.yatai.io/en/latest/index.html>`_.

For more information about the BentoML ecosystem, see the `BentoML
GitHub account <https://github.com/bentoml>`_.

.. note::
   The BentoML ecosystem extends beyond its core tools. It
   features seamless integrations with other popular technologies such
   as
   :doc:`MLFlow </integrations/mlflow>`,
   `LangChain <https://github.com/ssheng/BentoChain>`_,
   `Kubeflow <https://www.kubeflow.org/docs/external-add-ons/serving/bentoml/>`_,
   :doc:`Triton </integrations/triton>`,
   :doc:`Spark </integrations/spark>`, and
   :doc:`Ray </integrations/ray>`.
   This interoperability widens the reach of BentoML, fostering
   collaboration across various ecosystems and expanding opportunities
   for users of these tools.

See also
--------

- :doc:`/overview/what-is-bentoml`
- :doc:`/quickstarts/deploy-a-transformer-model-with-bentoml`
