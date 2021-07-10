Yatai Client
============

A Yatai RPC Server is a stateful service that provides a complete BentoML model management and model serving/deployment workflow.

Two sets of APIs are provided:

    - :class:`~bentoml.yatai.client.bento_repository_api.BentoRepositoryAPIClient` (via `YataiClient.repository`) manages saved :class:`~bentoml.BentoService` bundle, making them available for serving in production environments.

    - :class:`~bentoml.yatai.client.deployment_api.DeploymentAPIClient` (via `YataiClient.deployment`) deploys BentoServices to a variety of different cloud platforms, track deployment status, set up logging monitoring for your model serving workload.

.. note::

    We want to provide a better documentation on
    using :class:`~bentoml.yatai.client.deployment_api.DeploymentAPIClient`
    programmatically. For now refer to
    `deployment_api.py <https://github.com/bentoml/BentoML/blob/master/bentoml/yatai/client/deployment_api.py>`_
    or using the CLI commands :code:`bentoml deployment`

.. autofunction:: bentoml.yatai.client.get_yatai_client

.. autoclass:: bentoml.yatai.client.bento_repository_api.BentoRepositoryAPIClient
    :members:
    :exclude-members: download_to_directory

.. spelling::

    proto
    pb
    boolean
    IrisClassifier
    ci
    MyService
    gz