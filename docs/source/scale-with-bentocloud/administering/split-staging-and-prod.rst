=========================================
Split staging and production environments
=========================================

An organization is the workspace where your team manages all the resources, such as API keys, models, Bentos, and Deployments. Each organization is completely isolated from others. This means resources are not shared across organizations.

However, you can create dedicated organizations for different environments, such as staging and production, and enforce clear boundaries for security, access control, and resource management.

.. note::

    This feature is only available to Enterprise users. `Reach out to us <https://www.bentoml.com/contact>`_ if your team needs this feature.

Switch between organizations in the console
--------------------------------------------

If you have access to multiple organizations, you'll see a dropdown in the top bar of the BentoCloud console. Use it to switch between organizations. Each one displays only its own resources (models, Bentos, Deployments, API keys, etc.).

Switch between organizations in the BentoML CLI
------------------------------------------------

You can select your organization through contexts in the CLI. A context represents a specific login session associated with an organization.

Here are some common context commands:

.. code-block:: bash

    # Log in and create a context
    bentoml cloud login \
      --api-token 'my_token' \
      --endpoint 'https://my_org.cloud.bentoml.com' \
      --context staging

    # List all contexts you have
    bentoml cloud list-context

    # Show the current context
    bentoml cloud current-context

By default, BentoML commands use the current context. Use the ``--context`` option to run a command with a specific context:

.. code-block:: bash

    bentoml deploy . --context prod

To update the current context and use a new context as the default one:

.. code-block:: bash

    bentoml cloud update-current-context prod

This is useful in automation scripts, where you may want to set context once and reuse it.

Transfer resources between organizations
----------------------------------------

Since resources are isolated per organization, you cannot directly share objects between organizations. However, you can pull from one organization and push into another.

For example, you can transfer a model or Bento as follows. This ensures that exactly the same artifact tested in staging is promoted to production.

.. code-block:: bash

    # Pull a model from staging
    bentoml model pull my_model:v1 --context staging

    # Push a model to prod
    bentoml models push my_model:v1 --context prod

    # Pull a Bento from staging
    bentoml pull my_bento:v1 --context staging

    # Push a Bento to prod
    bentoml push my_bento:v1 --context prod

Automate deployments across organizations
-----------------------------------------

In real workflows, promotion between staging and production is often automated via CI/CD pipelines (e.g. GitHub Actions). BentoML supports different patterns depending on how you train and deploy.

Pattern 1: Training and deployment in a single pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    # deploy.sh
    CONTEXT="${CONTEXT:-staging}"

    python train.py

    # Optionally push the model explicitly
    # "bentoml deploy" will use local models when model reference set to "latest"
    # bentoml model push MY_MODEL:latest --context "$CONTEXT"

    bentoml deploy . --context "$CONTEXT"

- Running ``./deploy.sh`` will deploy to staging.
- Running ``CONTEXT=prod ./deploy.sh`` will deploy to production.

Pattern 2: Training and deployment in separate pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    # Training pipeline
    python train.py
    bentoml model push MY_MODEL:latest --context "$CONTEXT"

Here, training outputs the model, and deployment can run independently:

.. code-block:: bash

    bentoml deploy . --context "$CONTEXT"

-----------------------------------------

Depending on your process, you can adapt the above pipelines by:

- Pulling and pushing Bentos: Verify Deployment artifacts in staging, then push them into production for a consistent deployment.
- Using ``config-file.yaml``: Define :doc:`Deployment configuration </scale-with-bentocloud/deployment/configure-deployments>` in source control for reproducibility across environments.
- Parameterizing model versions: Use ``bentoml.use_arguments()`` to explicitly select which model version is deployed. See :doc:`template arguments </build-with-bentoml/template-arguments>` for details.