==================
Create Deployments
==================

Once you have fully tested your BentoML Service locally, you can push it to BentoCloud for production deployment. This document explains how to create a Deployment on BentoCloud.

Prerequisites
-------------

- Make sure you have logged in to BentoCloud using an :doc:`API token </bentocloud/how-tos/manage-access-token>` with Developer Operations Access.
- You have created a BentoML project that contains at least a ``service.py`` file and a ``bentofile.yaml`` file (or you have an available Bento either locally or on BentoCloud). You can use this :doc:`/get-started/quickstart` or any project in :doc:`/use-cases/index`.

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

If you already have a Bento, either available locally or on BentoCloud, you can use one of the following ways to deploy it.

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

    .. tab-item:: BentoCloud console

        The BentoCloud console provides a web-based, graphical user interface (UI) that you can use to create and manage your Bento Deployments. When you use the BentoCloud console to deploy a Bento, make sure the Bento is already available on BentoCloud.
