===========================
Configure standby instances
===========================

.. note::

   This feature is currently only available to BYOC users. Only users with the Admin role can configure standby instances.

Standby instances in BentoCloud allow you to prepare a set number of cloud machines in advance to handle potential demand surges. These instances are pre-provisioned by your cloud service provider (CSP), so they are ready to serve incoming requests immediately as your AI application scales. This helps minimize delays in serving new requests, especially during unexpected spikes.

The number of standby instances you set indicates the count of additional instances that BentoCloud will keep "ready" for your application. BentoCloud ensures this set number of standby instances always remains available, even as your application scales up or down.

Standby instances incur charges even when not actively serving deployments. This is because they are kept in a "ready" state to ensure quick scaling.

To configure standby instances:

1. Go to the **Clusters** section in the BentoCloud console.
2. Select the desired cluster (region) where you want to configure standby instances and click **Standby Instances**.
3. In the dialog that appears, set the desired number of standby instances for each instance type based on your anticipated demand.
4. Click **Submit**.
