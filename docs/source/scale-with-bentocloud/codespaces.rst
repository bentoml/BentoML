=======================
Develop with Codespaces
=======================

Codespaces is a development platform on BentoCloud designed to solve the challenges in building AI applications:

- Limited access to a variety of powerful GPUs in local development environments
- Slow iteration cycles due to container image building processes
- Inconsistent behavior between development and production environments

With Codespaces, you can connect your local development environment to BentoCloud while using your favorite IDE. This allows you to develop with powerful cloud GPUs in a reliable environment that is consistent with production. Your changes are updated in real-time on BentoCloud, as Codespaces automatically hot-reload when detecting local updates. This ensures you can iterate rapidly and confidently on your applications.

Create a Codespace
------------------

1. Navigate to the directory where your ``bentofile.yaml`` file is located. You can use the :doc:`Hello World project </get-started/hello-world>` as an example.
2. Run the following command to create a Codespace:

   .. code-block:: bash

      bentoml code

3. (Optional) If your project requires secure environment variables like API keys, we recommend you create :doc:`secrets </scale-with-bentocloud/manage-secrets-and-env-vars>` for them. For example:

   .. code-block:: bash

      bentoml secret create huggingface HF_TOKEN=<your_hf_token>
      bentoml code --secret huggingface

4. Follow the on-screen instructions to create a new Codespace (or attach to an existing one) as prompted. Once created, you can view it in the **Codespaces** section of BentoCloud.

   .. image:: ../../_static/img/bentocloud/how-to/codespaces/codespace-on-bentocloud.png

Test your application
---------------------

Once the Codespace is up and running, you can test your application by calling its exposed endpoint.

While developing with your Codespace, the following changes will be automatically synchronized between your local environment and the Codespace. This means any local code updates will be reflected in the remote Codespace, automatically triggering a reload.

- BentoML Service code, including adding or deleting Services
- Models
- System packages
- Python requirements

Codespace logs will stream directly to your terminal, giving you real-time feedback on your application's performance. You can also debug the application using the provided endpoint URL on the Codespace details page.

Note that the following changes cannot be synchronized:

- Python version (Codespaces currently use Python 3.11)
- Base image and Docker options

Build your Bento
----------------

Once development is complete, you can build your Bento directly from a Codespace. To do this, click the **Build Bento** button.

.. image:: ../../_static/img/bentocloud/how-to/codespaces/codespace-detail-page.png

After the build is successful, you can view the Bento and create a Deployment with it.

After development
-----------------

To exit the Codespace from your terminal, press ``Ctrl+C``. Note that this **DOES NOT** terminate the remote Codespace automatically. To terminate a remote Codespace, run the following command:

.. code-block:: bash

   bentoml deployment terminate <codespace_name>

Reattach to an existing Codespace
---------------------------------

To reattach to a previously created Codespace, use the following command:

.. code-block:: bash

   bentoml code --attach <codespace_name>

This will synchronize the remote Codespace with your current local code.
