=====================
Build CI/CD pipelines
=====================

This document explains how to build CI/CD pipelines to automate the deployment of BentoML :doc:`Services </build-with-bentoml/services>` on BentoCloud using GitHub Actions.

Repository structure
--------------------

Organize your GitHub repository as follows and here is an `example <https://github.com/bentoml/bentocloud-cicd-example>`_:

.. code-block:: bash

    bentocloud-cicd-example/
    ├── service.py             # Your BentoML Service
    ├── deployment.yaml        # Deployment config on BentoCloud
    ├── requirements.txt       # (Optional) Python dependencies
    ├── README.md              # Project documentation
    └── .github/
        └── workflows/
            ├── code-update.yml         # Handles code or general changes
            └── deployment-config.yml   # Handles Deployment config changes

Prerequisites
-------------

- You must have a BentoCloud account. `Sign up here <https://cloud.bentoml.com/signup>`_ if you don't have one yet.
- Create an :doc:`API Token </scale-with-bentocloud/manage-api-tokens>`. For production pipelines, we recommend Organization API tokens to ensure continuity.
- In your GitHub repository, go to **Settings > Secrets and variables > Actions > Secrets** and add the following:

  - ``BENTOCLOUD_API_TOKEN``: Your BentoCloud API token.
  - ``DEPLOYMENT_NAME``: The name of your Deployment on BentoCloud.
  - ``DEPLOYMENT_URL``: The endpoint URL of your Deployment.
  - ``BENTOML_VERSION``: (Optional) The BentoML version to install. It defaults to the latest version.

GitHub Actions
--------------

This example contains two workflows. Once triggered, they:

1. Set up the environment and install dependencies
2. Update your existing BentoML Deployment
3. Wait until the Deployment is ready
4. Perform a test inference

``code-update.yml``
^^^^^^^^^^^^^^^^^^^

This workflow triggers on any change to the ``main`` branch except ``deployment.yaml``. You can use it to handle general code changes:

.. code-block:: yaml
    :caption: `code-update.yml`

    name: Update Code

    on:
      workflow_dispatch:
      push:
        branches:
          - main
        paths-ignore:
          - 'deployment.yaml'

    jobs:
      update-code:
        runs-on: ubuntu-latest

        env:
          BENTOML_VERSION: ${{ secrets.BENTOML_VERSION }}

        steps:
          - name: Checkout Code
            uses: actions/checkout@v3

          - name: Set up Python
            uses: actions/setup-python@v4
            with:
              python-version: '3.11'

          - name: Install BentoML
            run: |
              python -m pip install --upgrade pip
              if [ -z "$BENTOML_VERSION" ]; then
                pip install bentoml
              else
                pip install "bentoml==$BENTOML_VERSION"
              fi
              pip install huggingface_hub

          - name: Log in to BentoCloud
            run: |
              echo "Logging in to BentoCloud"
              bentoml cloud login --api-token ${{ secrets.BENTOCLOUD_API_TOKEN }}

          - name: Deploy and Run Test Inference
            shell: python
            run: |
              import bentoml

              deployment = bentoml.deployment.update(
                  name="${{ secrets.DEPLOYMENT_NAME }}",
                  bento=".",
              )

              code = deployment.wait_until_ready(timeout=60)
              if code != 0:
                  raise RuntimeError("Deployment did not become ready in time.")

              client = deployment.get_client()
              response = client.summarize(text="This is a sample prompt for testing.")
              print(response)

``deployment-config.yml``
^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow is triggered only when ``deployment.yaml`` changes, which contains Deployment configurations on BentoCloud, such as scaling replicas, GPU instance types, and update strategy. You only need to specify the fields you want to update `in the file <https://github.com/bentoml/bentocloud-cicd-example/blob/main/deployment.yaml>`_.

.. code-block:: yaml
    :caption: `deployment-config.yml`

    name: Update Deployment Config

    on:
      workflow_dispatch:
      push:
        branches:
          - main
        paths:
          - 'deployment.yaml'

    jobs:
      update-deployment-config:
        runs-on: ubuntu-latest

        env:
          BENTOML_VERSION: ${{ secrets.BENTOML_VERSION }}

        steps:
          - name: Checkout Code
            uses: actions/checkout@v3

          - name: Set up Python
            uses: actions/setup-python@v4
            with:
              python-version: '3.11'

          - name: Install BentoML
            run: |
              python -m pip install --upgrade pip
              if [ -z "$BENTOML_VERSION" ]; then
                pip install bentoml
              else
                pip install "bentoml==$BENTOML_VERSION"
              fi

          - name: Log in to BentoCloud
            run: |
              echo "Logging in to BentoCloud"
              bentoml cloud login --api-token ${{ secrets.BENTOCLOUD_API_TOKEN }}

          - name: Deploy and Run Test Inference
            shell: python
            run: |
              import bentoml

              deployment = bentoml.deployment.update(
                  name="${{ secrets.DEPLOYMENT_NAME }}",
                  config_file="deployment.yaml",
              )

              code = deployment.wait_until_ready(timeout=60)
              if code != 0:
                  raise RuntimeError("Deployment did not become ready in time.")

              client = deployment.get_client()
              response = client.summarize(text="This is a sample prompt for testing.")
              print(response)

Build CI/CD pipelines for ML projects
-------------------------------------

In ML projects, models often change more frequently than code, typically due to new training data or parameter tuning.

Once your new model is :doc:`saved and pushed to BentoCloud </build-with-bentoml/model-loading-and-management>`, you need to trigger a deployment workflow. However, if your Service code or Deployment configuration hasn't changed, GitHub Actions won't trigger automatically in the above examples.

Here are common strategies for triggering CI/CD workflows based on model updates:

- **Manually trigger the workflow in GitHub**. This avoids complexity and works well for lightweight use cases, where:

  - You're always using the ``latest`` model version (e.g. ``model_name:latest``)
  - You only manage a single model/deployment
  - Model updates are infrequent or can be manually verified

- **Commit a model metadata file**. For more flexible and automated workflows:

  - Track the model tag in a file (e.g., ``model_tag.txt``)
  - Use :doc:`template arguments </build-with-bentoml/template-arguments>` or environment variables to pass the model tag to your Service
  - Trigger your workflow on commits to that file

  This provides fine-grained control for managing different models, versions, or Deployment targets.

- **Call the GitHub API to trigger a workflow**. This lets you initiate a workflow from external systems like an ML training pipeline (e.g., Airflow or custom scripts). It provides full automation (from training → model registration → deployment), and supports versioned or conditional deployments, without needing to commit code. For more information, see `the GitHub API documentation <https://docs.github.com/en/rest?apiVersion=2022-11-28>`_.

More resources
--------------

- :doc:`manage-deployments`
- :doc:`call-deployment-endpoints`
