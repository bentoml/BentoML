==============
GitHub Actions
==============

BentoML provides a GitHub Action to help you automate the process of building Bentos and deploying them to the cloud.
To create a GitHub Actions workflow, you need to first define a workflow file as below:

.. literalinclude:: ./../data/workflow.yaml
   :language: yaml

The action `bentoml/deploy-bento-action <https://github.com/bentoml/deploy-bento-action>`_ requires ``cloud_api_token`` and ``cloud_endpoint``. You can set the `repository secrets <https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository>`_ to make sure the job has correct privilege.

.. seealso::

   You can get more details about this file in :ref:`Deploying Your Bento <bentocloud/how-tos/deploy-bentos:Deploying Your Bento>` and `the GitHub Actions documentation <https://docs.github.com/en/actions/quickstart>`_.

With this workflow, every time you push changes to the repository, a new Bento will be built and rolled out to the existing deployment.

Read the usage and available input parameters of this action in the `Action README <https://github.com/bentoml/deploy-bento-action>`_.
