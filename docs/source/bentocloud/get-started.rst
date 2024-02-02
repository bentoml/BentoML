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

Plans
-----

BentoCloud is available with the following two plans.

Starter
^^^^^^^

The Starter plan is designed for small teams of developers who want to focus on building AI applications without infrastructure management. With the autoscaling feature of BentoCloud, you only pay for the resources you use.

Enterprise
^^^^^^^^^^

The Enterprise plan includes all the features offered in the Starter plan. It is tailored for teams that want to use BentoCloud in :doc:`their own cloud or on-premises environment (BYOC) </bentocloud/how-tos/byoc>`, ensuring data security and compliance. If you prefer not to use your own cluster, we can provide a dedicated cloud environment for you. Either way, we take care of managing the infrastructure to ensure a scalable and secure model deployment experience.

Access BentoCloud
-----------------

To gain access to BentoCloud, sign up here:

.. raw:: html

    <a href="https://kdyvd8c5ifq.typeform.com/to/eTujPAaE" class="custom-button demo">Schedule a Demo</a>
    <a href="https://cloud.bentoml.com" class="custom-button trial">Start Free Trial</a>

Once you have your BentoCloud account, do the following to get started:

1. Install BentoML by running ``pip install bentoml``. See :doc:`/get-started/installation` for details.
2. Create an :doc:`API token with Developer Operations Access </bentocloud/how-tos/manage-access-token>`.
3. Log in to BentoCloud with the ``bentoml cloud login`` command, which will be displayed on the BentoCloud console after you create the API token.

Now, you can try an `example project and deploy it to BentoCloud <https://github.com/bentoml/quickstart>`_.

Resources
---------

If you are a first-time user of BentoCloud, we recommend you read the following documents to get familiar with BentoCloud:

- Deploy :doc:`example projects </use-cases/index>` to BentoCloud
- :doc:`/bentocloud/how-tos/manage-deployments`
