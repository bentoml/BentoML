==================
Create Deployments
==================

Once you have fully tested your BentoML Service locally, you can push it to BentoCloud for production deployment. This document explains how to create a Deployment on BentoCloud.

Prerequisites
-------------

- Make sure you have logged in to BentoCloud using an :doc:`API token </scale-with-bentocloud/manage-api-tokens>` with Developer Operations Access.
- You have created a BentoML project that contains at least a ``service.py`` file and a ``bentofile.yaml`` file (or you have an available Bento either locally or on BentoCloud). You can use this :doc:`/get-started/hello-world` or any project in :doc:`Examples </examples/overview>`.

Deploy a new project to BentoCloud
----------------------------------

You can deploy a new project through either the command line interface (CLI) or Python API.

.. tab-set::

    .. tab-item:: BentoML CLI

        In your project directory where the ``bentofile.yaml`` file is stored, run the following command and use the ``-n`` flag to optionally set a name.

        .. code-block:: bash

           bentoml deploy . -n <deployment_name>

        You can see the Deployment status in your terminal after running the command.

    .. tab-item:: Python API

        Specify the path to your BentoML project using the ``bento`` parameter and optionally set a name.

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(bento = "./path_to_your_project", name = "my_deployment_name")

        You can use the block function ``wait_until_ready`` to periodically check the status of a Deployment until it becomes ready or until a specified timeout is reached.

        .. code-block:: python

            import bentoml

            dep = bentoml.deployment.create(bento="./path_to_your_project", name = "my_deployment_name")

            dep.wait_until_ready(timeout=3600)

BentoML does the following automatically during deployment:

1. **Build**: Build your project into a Bento based on ``bentofile.yaml``.
2. **Push**: Push the Bento to BentoCloud.
3. **Deploy**: Deploy the Bento on BentoCloud by performing the following steps in order:

   a. Containerize the Bento as an OCI-compliant image.
   b. Provision instances on BentoCloud.
   c. Start the BentoML Service on the instances based on the specified configuration.

.. note::

   You **DO NOT** need to perform the above three steps (Build, Push, and Deploy) manually, which is a long-running automated process.

Deploy an existing Bento to BentoCloud
--------------------------------------

If you already have a Bento built locally (run ``bentoml list`` to view all the local Bentos), you can deploy it using either the BentoML CLI or Python API.

.. tab-set::

    .. tab-item:: BentoML CLI

        In your project directory where the ``bentofile.yaml`` file is stored, run the following command and use the ``-n`` flag to optionally set a name.

        .. code-block:: bash

            bentoml deploy bento_name:version -n <deployment_name>

    .. tab-item:: Python API

        Specify the Bento tag using the ``bento`` parameter and optionally set a name.

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(bento = "bento_name:version", name = "my_deployment_name")

        You can use the block function ``wait_until_ready`` to periodically check the status of a Deployment until it becomes ready or until a specified timeout is reached.

        .. code-block:: python

            import bentoml

            dep = bentoml.deployment.create(bento = "bento_name:version", name = "my_deployment_name")

            dep.wait_until_ready(timeout=3600)

The ``bentoml deploy`` command and the ``bentoml.deployment.create`` function automatically push and deploy the Bento to BentoCloud. If you only need to share a Bento with your team and deploy it later, you can push the Bento to BentoCloud by running the following command:

.. code-block:: bash

    $ bentoml push <bento_name:version>

    ╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ Successfully pushed Bento "bento_name:version"                                                                                                                                   │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    Pushing Bento "bento_name:version" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% • 51.2/51.2 kB • ? • 0:00:00

You can then view your pushed Bento on the BentoCloud console, which provides a web-based, graphical user interface (UI), and create a Deployment using the Bento.
