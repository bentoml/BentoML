==========
Deployment
==========

Once you have fully tested your BentoML Service locally, you can push it to :doc:`BentoCloud </bentocloud/get-started>` with a single command for deployment. BentoCloud is a fully-managed platform designed for running AI applications. It provides comprehensive solutions to deployment, scalability, and collaboration challenges in the AI application delivery lifecycle. As BentoCloud manages the underlying infrastructure for you, you only need to focus on developing AI applications.

If you want to deploy your BentoML project in environments other than BentoCloud, you can manually build a Bento for your project, containerize it, and deploy it to any Docker-compatible environments. See :doc:`/get-started/packaging-for-deployment` for details.

To deploy a BentoML project on BentoCloud, do the following:

1. `Sign up here <https://www.bentoml.com/cloud>`_ for BentoCloud. You will receive $10 free credits once you have your account approved.
2. Run the ``bentoml cloud login`` command and follow the on-screen instructions to log in.
3. Deploy a project on BentoCloud. You can deploy the example project in :doc:`/get-started/hello-world` by running the following commands:

   .. code-block:: bash

      git clone https://github.com/bentoml/quickstart.git
      cd quickstart
      pip install -r requirements.txt
      bentoml deploy .

4. You can then interact with the Deployment on the console.

   .. image:: ../../_static/img/guides/deployment/deployment-replica.png

It is possible to customize deployment configurations like scaling and instance type. For details, see :doc:`/scale-with-bentocloud/deployment/configure-deployments`.
