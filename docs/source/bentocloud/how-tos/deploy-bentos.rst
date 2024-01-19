=============
Deploy Bentos
=============

Deploying a machine learning model can be quite a task. However, BentoCloud simplifies the process by enabling you to serve your model as an online service or on-demand function. This guide will walk you through how to deploy a Bento on BentoCloud.

Deployment mode
===============

There are two Deployment modes on BentoCloud:

Online Service
--------------

The Online Service Deployment offered by BentoCloud is ideal for low-latency serving scenarios. To ensure requests can be promptly addressed, instances in this mode are never scaled down to zero, maintaining a ready state for immediate processing. Furthermore, requests are directly routed to the serving instances of the API Server and Runners, bypassing any queuing mechanisms. This direct routing mechanism ensures minimum latency, providing an efficient and swift response to incoming requests.

On-Demand Function
------------------

The On-Demand Function Deployment offered by BentoCloud is particularly suited for situations that prioritize cost-efficiency and reliability. In scenarios where requests are sporadic, this mode enables instances to scale down to zero, thereby conserving resources. This feature proves particularly beneficial for GPU-accelerated instances, which are generally more expensive to maintain. To ensure maximum reliability, especially during periods of cold-start or overload, requests are queued prior to processing. This mechanism enables the system to handle bursts of requests effectively, thus enhancing the robustness and dependability of your application under varying load conditions.

Build your Bento
================

1. To build your machine learning application into a Bento, check out this :doc:`/concepts/bento` in BentoML’s doc.
2. To push your Bento to BentoCloud, do ``bentoml push <name>:<tag>``.  See :doc:`manage-models-and-bentos` for more details.

Manage Deployments with the BentoCloud Console
==============================================

The BentoCloud Console enables you to perform basic management tasks with your Bentos using a browser.

Deploy your Bento
-----------------

1. Navigate to the **Deployments** section on BentoCloud and click the **Create** button in the upper-right corner.
2. Choose the Deployment type (**Online Service** or **On-Demand Function**).

   .. image:: ../../_static/img/bentocloud/type-of-deployment.png

3. Name your Deployment, select the Bento you want to deploy, and specify other details like the number of instances, the amount of memory, and more. Note that if your desired Repository or Bento is not displayed in the drop-down list, enter its name to search for it.

   .. image:: ../../_static/img/bentocloud/create-deployment.png

4. Click **Submit**.

Under the hood, the Bento is being built into an OCI Image to be deployed in BentoCloud. The deployment might take a few minutes, depending on your configuration.

View your Deployment
--------------------

After your Bento is deployed, do the following to check the status of the Deployment:

1. Navigate to the `Deployments <http://cloud.bentoml.com/deployment>`_ section.
2. Select the desired Deployment. On the Deployment details page, view the Deployment information, such as status, events, replicas, and revisions.

   .. image:: ../../_static/img/bentocloud/viewing-deployment.png

3. To update the Deployment, click **Update** in the upper-right corner, update your desired field, and click **Submit**.

Terminate your Deployment
-------------------------

You can temporarily stop a Bento Deployment to make its endpoint inaccessible. The terminated Deployment can be restarted later and all the revision records are preserved.

To terminate a Deployment, do the following:

1. On the Deployment details page, click **Terminate** in the upper-right corner.
2. In the dialog that appears, enter the Deployment's name.
3. Click **Terminate**.

You can restart the Deployment to make it available again by clicking **Restore**.

Delete your Deployment
----------------------

After a Deployment is terminated, you can delete it. All the revision records will be deleted as well.

To delete a Deployment, do the following:

1. On the Deployment details page, click **Delete** in the upper-right corner.
2. In the dialog that appears, enter the Deployment's name.
3. Click **Delete**.

.. warning::

   You can't recover a Deployment after deleting it. This action is irreversible.

Manage Deployments with the BentoML CLI
=========================================

The BentoML CLI is a set of tools that you can use to deploy any machine learning models as production-ready API endpoints on the cloud.
To create and manage your Bento Deployments on BentoCloud, use ``bentoml deployment`` with the corresponding options.

.. _cli-deploy-bento:

Deploy your Bento
-----------------

Currently, the BentoML CLI only supports creating and updating a Bento Deployment by specifying a JSON file, which contains detailed configurations of the Deployment, such as ``name``, ``mode``, and ``targets``.
The JSON file follows the same syntax as the **JSON** tab when you create or update a Deployment on the BentoCloud Console.

Run the following command to deploy a Bento.

.. tab-set::

    .. tab-item:: Using an existing JSON file

        .. code-block:: bash

          bentoml deployment create --file <file_name>.json

    .. tab-item:: Using a HereDoc

        .. code-block:: bash

          bentoml deployment create --file - <<EOF
          {
              "name": "deployment-name",
              "cluster_name": "default",
              "description": "My first Deployment.",
              "mode": "function",
              "targets": [
                  {
                      "type": "stable",
                      "bento_repository": "iris_classifier",
                      "bento": "3auspqat6smdonry",
                      "config": {
                          "hpa_conf": {
                              "min_replicas": 1,
                              "max_replicas": 2,
                      ...
          }
          EOF

View your Deployment
--------------------

Run the following command to view all the existing Deployments on BentoCloud:

.. code-block:: bash

   bentoml deployment list

Run the following command to view the detailed information about a specific Bento Deployment:

.. code-block:: bash

   bentoml deployment get <deployment_name>

Run the following command to update a Deployment.

.. tab-set::

    .. tab-item:: Using an existing JSON file

        .. code-block:: bash

          bentoml deployment update --file <file_name>.json

    .. tab-item:: Using a HereDoc

        .. code-block:: bash

          bentoml deployment update --file - <<EOF
          {
              "name": "deployment-name",
              "cluster_name": "default",
              "description": "My first Deployment.",
              "mode": "function",
              "targets": [
                  {
                      "type": "stable",
                      "bento_repository": "iris_classifier",
                      "bento": "3auspqat6smdonry",
                      "config": {
                          "hpa_conf": {
                              "min_replicas": 1,
                              "max_replicas": 3,
                      ...
          }
          EOF

Terminate your Deployment
-------------------------

You can temporarily stop a Bento Deployment to make its endpoint inaccessible. The terminated Deployment can be restarted later and all the revision records are preserved.

Run the following command to terminate a Deployment.

.. code-block:: bash

   bentoml deployment terminate <deployment_name>

Delete your Deployment
----------------------

After a Deployment is terminated, you can delete it. All the revision records will be deleted as well.

Run the following command to delete a Deployment.

.. code-block:: bash

   bentoml deployment delete <deployment_name>

.. warning::

   You can't recover a Deployment after deleting it. This action is irreversible.

For more information about ``bentoml deployment``, see :doc:`/reference/cli`.

Interact with your Deployment
=============================

Now that your model is deployed, you can send requests to it. Here's an example of how to send a request to your deployed model using ``curl``:

For HTTP-based servers, include the token in the header of your HTTP request like this:

.. code-block:: bash

   curl "http://flan.bentocloud.com/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $YOUR_TOKEN" \
     --data '{"prompt": "What state is Los Angeles in?", "llm_config": {"max_new_tokens": 129}}'

The exact way you interact with your Deployment will depend on the :doc:`Service </concepts/service>`
endpoints and the :ref:`io-descriptors` of the endpoint.
