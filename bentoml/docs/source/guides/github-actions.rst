==============
GitHub Actions
==============

`GitHub Actions <https://github.com/features/actions>`_ is a continuous integration and continuous delivery (CI/CD) platform that allows you to automate your build, test, and deployment pipeline.
You can create various workflows to accelerate and simplify the BentoML workflow by automating important operations, such as building and deploying Bentos.

Set up BentoML
--------------

You can create a GitHub Action to bootstrap BentoML and any necessary supporting tools to run BentoML in a CI process.

To use it, define a workflow file as below:

.. literalinclude:: ./../data/bentoml-setup.yaml
   :language: yaml

Once triggered, this workflow sets up BentoML in an Ubuntu environment, ensuring that it's using Python 3.10 and the ``main`` branch version of BentoML.
For more information, see the `setup-bentoml-action repository <https://github.com/bentoml/setup-bentoml-action>`_.

Deploy Bentos to the cloud
--------------------------

You can create a GitHub Actions workflow to automate the process of building Bentos and deploying them to the cloud.

To use it, define a workflow file as below:

.. literalinclude:: ./../data/deploy-bento-to-cloud.yaml
   :language: yaml

The workflow `bentoml/deploy-bento-action <https://github.com/bentoml/deploy-bento-action>`_ requires ``cloud_api_token`` and ``cloud_endpoint``. You can set the `repository secrets <https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository>`_ to make sure the job has correct privilege.

.. seealso::

   You can get more details about this file in :ref:`Deploying Your Bento <bentocloud/how-tos/deploy-bentos:Deploying Your Bento>` and `the GitHub Actions documentation <https://docs.github.com/en/actions/quickstart>`_.

With this workflow, every time you push changes to the repository, a new Bento will be built and rolled out to the existing deployment.

Read the usage and available input parameters of this workflow in the `deploy-bento-action GitHub repository <https://github.com/bentoml/deploy-bento-action>`_.

Build Bentos from a GitHub repository
-------------------------------------

You can create a GitHub Action workflow to automate the process of building Bentos either from a GitHub repository or from a specified context path.

To use it, define a workflow file as below:

.. literalinclude:: ./../data/build-bento-action.yaml
   :language: yaml

This workflow first uses the `BentoML setup workflow <https://github.com/bentoml/setup-bentoml-action>`_ to prepare the necessary environment and
then builds the Bento using the default GitHub context. To specify a different context, set the ``context`` path:


.. code-block:: yaml

   - uses: bentoml/build-bento-action@v1
     with:
       context: 'path/to/context'

To specify a version, use the ``version`` field:

.. code-block:: yaml

   - uses: bentoml/build-bento-action@v1
     with:
       version: '1.0.0'

To specify a different ``bentofile.yaml`` file, use the ``bentofile`` field:

.. code-block:: yaml

   - uses: bentoml/build-bento-action@v1
     with:
       bentofile: 'path/to/bentofile'

.. note::

   ``bentofile`` must be a path relative to the ``context`` directory.

For more information, see the `build-bento-action repository <https://github.com/bentoml/build-bento-action>`_.

Create and push Bento images to a container registry
----------------------------------------------------

You can create a GitHub Actions workflow to create and push Bento images to any container registry (Docker Hub, GHCR, ECR, GCR, etc.). This workflow uses `Docker Buildx <https://github.com/docker/buildx>`_,
which offers extended capabilities for building containers and is enhanced by the `Moby BuildKit builder toolkit <https://github.com/moby/buildkit>`_.

To use it, define a workflow file as below:

.. literalinclude:: ./../data/containerize-and-push.yaml
   :language: yaml

In this example, the workflow first uses the `BentoML setup workflow <https://github.com/bentoml/setup-bentoml-action>`_ to prepare the necessary environment. It then
sets up Docker QEMU and Docker Buildx, and configures Docker Hub authentication. Lastly, it builds a Bento, creates a Docker image, and pushes it to Docker Hub.

.. note::

   The workflow is essentially an adaptation of Docker's `build-push-action <https://github.com/docker/build-push-action>`_,
   specifically tailored for implementing Bento containerization. You can also use ``docker/login-action@v2`` to log in to other container registries
   supported by that action. Refer to `the login action <https://github.com/docker/login-action?tab=readme-ov-file#usage>`_ for more information.
