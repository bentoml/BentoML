=========================
Create canary Deployments
=========================

Rolling out new versions of AI services requires careful planning. If you immediately shift all traffic to a new Deployment, it can introduce latency, regressions, or performance issues as your Deployment scales to meet demand. These risks are amplified under high traffic conditions, which can directly disrupt the user experience.

Canary Deployments on BentoCloud help mitigate these risks by enabling you to:

- Deploy multiple Bento versions simultaneously
- Choose from multiple routing strategies for fine-grained control
- Gradually shift traffic between versions (e.g., 90% to stable, 10% to new)
- Implement rapid rollbacks with minimal risk
- Monitor real-time performance across deployed versions

.. image:: ../../_static/img/bentocloud/how-to/canary-deployments/carnary-deployment-bentocloud.png
   :alt: Canary Deployment on BentoCloud
   :width: 65%
   :align: center

How to use canary Deployments
-----------------------------

1. On the **Configuration** page, toggle on **Canary** mode when creating or updating a Deployment.

   .. image:: ../../_static/img/bentocloud/how-to/canary-deployments/canary-config.png
      :alt: Enable canary model on BentoCloud

2. Select one of the following traffic routing strategies.

   - **Split traffic by header**: Hash a specified HTTP header to route traffic consistently to the same version. This is ideal for sticky sessions or user-based routing.

     Example:

     - Header/Query Key: ``X-User-ID``
     - Client request:

       .. code-block:: bash

          curl -H "X-User-ID: user123" ...

   - **Split traffic by query parameter**: Hash a query parameter in the URL to determine routing.

     Example:

     - Header/Query Key: ``feature``
     - Client request:

       .. code-block:: bash

          curl "http://your-endpoint-url/predict?feature=test" ...

   - **Random**: Distribute traffic randomly between versions according to the specified percentages.

3. Select the Bento versions to include and assign traffic percentages to each. For example:

   - Bento v1: 10%
   - Bento v2: 30%
   - Remaining 60% will go to the default Bento version.

   .. note::

      Total traffic allocation across all versions must not exceed ``100%``.

4. Once the configuration is saved, navigate to the **Playground** tab to test each version independently using the version selector.

   .. image:: ../../_static/img/bentocloud/how-to/canary-deployments/multiple-bento-versions.png
      :alt: Test different Bento versions

5. Use the **Monitoring** tab to view real-time performance metrics for each Bento version in the canary Deployment.

   .. image:: ../../_static/img/bentocloud/how-to/canary-deployments/canary-performance.png
      :alt: Test different Bento versions

6. Once you're confident in a version's performance, simply edit the Deployment and increase its traffic share to 100%.

.. note::

   You can also configure canary Deployments programmatically using the BentoML CLI or Python SDK. Define your canary Deployment using the ``canary`` field in a configuration file. View the full equivalent code on the **Configuration** page of the BentoCloud console.
