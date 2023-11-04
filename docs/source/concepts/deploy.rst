==========
Deployment
==========

As the standard distribution format in the BentoML ecosystem, :doc:`/concepts/bento` can be deployed in different ways. In essence, these deployment strategies rely on Bento containerization underneath.

This page explains different Bento deployment strategies.

BentoCloud
----------

BentoCloud is a fully-managed platform designed for building and operating AI applications. It provides comprehensive solutions for addressing deployment, scalability, and collaboration challenges in the AI application delivery lifecycle.
As BentoCloud manages the underlying infrastructure for you, you only need to focus on developing AI applications. BentoCloud is currently available for early access with two plans - Starter and Enterprise. See the :doc:`BentoCloud documentation </bentocloud/getting-started/understand-bentocloud>` to learn more.

To deploy a Bento on BentoCloud:

1. Create an :doc:`API token </bentocloud/how-tos/manage-access-token>` with Developer Operations Access on BentoCloud.
2. Log in to BentoCloud with the token.
3. Push the Bento to BentoCloud using ``bentoml push``.
4. Deploy the Bento via the BentoCloud console. Alternatively, create a Deployment configuration file in JSON and use the BentoML CLI (``bentoml deployment create --file <file_name>.json``) to deploy it.

For details, see :doc:`/bentocloud/getting-started/quickstart` and :doc:`/bentocloud/how-tos/deploy-bentos`.

Docker
------

When a Bento is built, BentoML automatically creates a Dockerfile within the Bento. This allows you to containerize the Bento as a Docker image, which is useful for
testing out the Bento's environment and dependency configurations locally.

To containerize a Bento:

1. Make sure you have `installed Docker <https://docs.docker.com/engine/install/>`_.
2. Run ``bentoml containerize BENTO_TAG`` to start the containerization process. You can use ``bentoml list`` to view available Bentos locally.

   .. note::

      If you are using Mac computers with Apple silicon, you can specify the ``--platform`` option to avoid potential compatibility issues with some Python libraries.

      .. code-block:: bash

          bentoml containerize --opt platform=linux/amd64 BENTO_TAG

3. View the built Docker image by running ``docker images``.
4. Run the generated Docker image by running ``docker run -p 3000:3000 IMAGE_TAG``. Note that ``3000`` is the default port for the Bento server.

.. seealso::

   Starting from version 1.0.11, BentoML supports `multiple container engines <https://github.com/bentoml/BentoML/pull/3164>`_ in addition to Docker. See :ref:`guides/containerization:Containerization with different container engines` about more details on the containerization process.

With the image ready, you can deploy it to any Docker-compatible environments, including but not limited to the following:

- Container orchestration systems: Kubernetes, Docker Swarm, Red Hat OpenShift, and Nomad.
- Container management services: Amazon ECS, Azure Container Instances, Google Cloud Run, and Apache Mesos.
- Yatai: `Yatai <https://github.com/bentoml/Yatai>`_ is the open-source Kubernetes deployment operator for BentoML. DevOps teams can seamlessly integrate BentoML into their GitOps workflow to deploy and scale ML services on Kubernetes.
  Yatai contains a subset of scalability features offered by BentoCloud.

.. important::

   When using the above strategies for production deployment, we recommend you consider the following factors for better performance, scalability, observability, resource utilization, and cost efficiency.

   - :doc:`/bentocloud/best-practices/cost-optimization`
   - :doc:`/guides/tracing`
   - :doc:`/guides/containerization`
   - :doc:`/guides/scheduling`
   - :doc:`/guides/gpu`
   - :doc:`/bentocloud/best-practices/bento-building-and-deployment`
