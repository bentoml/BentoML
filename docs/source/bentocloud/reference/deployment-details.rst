==================
Deployment details
==================

Deployment details refer to the properties of a Bento Deployment, such as its metadata, status, monitoring metrics, and revision records.
You set some of these properties when you create the Deployment, and you also have the option to edit some of them as needed after creation.

Overview
--------

Overview
^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Name
     - The name of the Deployment, serving as a unique identifier for the Bento Deployment on BentoCloud.
   * - URL
     - The endpoint at which the Bento Deployment is accessible.
   * - Bento
     - The specific Bento package being used for this Deployment, which includes the model, source code, and dependencies.
   * - Endpoint Access Type
     - The level of access allowed to the endpoint, including Public and Protected.
   * - Description
     - A brief explanation of the Bento Deployment.
   * - Creator
     - The user who created this Bento Deployment.
   * - Created at
     - The date and time when the Bento Deployment was initially created.

Component Status
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Name
     - The name of the component, such as API Server or a specific Runner.
   * - Instance Type
     - The classification of the cloud instance that the component is using.
   * - Resources
     - The allocated computing resources for the component, such as CPU and memory.
   * - Status
     - The current operational status of the component, indicated by one of the status lights:

       - Green: Active
       - Grey: Not deployed
       - Red: Failed

Events
^^^^^^

This section displays a chronological record of system events related to the Deployment's operation on BentoCloud. Each entry logs the actions performed by the system, such as scaling, component status changes, and updates to the Bento configurations. These logs are instrumental for auditing, debugging, and understanding the behavior of the Deployment over time.

You can search for event keywords in the search bar.

.. note::

   The Events section is also available on the **Replicas** and **Revisions** tabs.

Replicas
--------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Group
     - The group type the replica belongs to, including API Server and Runner.
   * - Name
     - The name of the API Server or Runner replica.
   * - Status
     - The status of the replica, indicating whether the replica is running or if it needs your attention. Possible statuses include ``Running`` and ``Failed``.
   * - Cluster
     - The name of the cloud cluster where the replica is running.
   * - Node
     - The specific node within the cluster where the replica is running.
   * - Start Time
     - The time when the replica was deployed.
   * - Operation
     - Operations that you can perform:

       - View logs: Inspect the logs of different containers within the Pod replica. You can use the drop-down menu at the top to switch between containers.
       - Inspect events: View the operational events of the replica.
       - Enter containers: `Exec into different containers <https://kubernetes.io/docs/tasks/debug/debug-application/get-shell-running-container/>`_ of the Pod replica. You can click **Show File Manager** to upload files to and download files from the container.
       - Troubleshoot containers: `Troubleshoot issues with an ephemeral debug container <https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/#ephemeral-container>`_. You can click **Show File Manager** to upload files to and download files from the container.

Logging
-------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Advanced Search
     - Search for logs using advanced filters. It supports regular expressions.
   * - Component
     - The component whose logs need to be displayed.
   * - Max Lines
     - The maximum number of log entries displayed.
   * - Logs volume
     - A histogram that displays the volume of log entries over a selected period, providing a visual representation of the activity levels and potential anomalies related to the Deployment’s operation.
   * - Time
     - The timestamp of when each log entry is recorded.
   * - Unique labels
     - Labels that provide context for the logs, such as the node, cluster, and Deployment name.
   * - Wrap lines
     - Wrap long lines for easier readability.
   * - Prettify JSON
     - Formats JSON log entries to be more human-readable.
   * - Deduplication
     - Filters out duplicate log entries. Options including "None", "Exact", "Numbers" and "Signature" provide different methods to identify and remove repeated logs.
   * - Display results
     - Sort log entries chronologically, either from the newest or the oldest.
   * - Download
     - Download logs in text or JSON format.

.. note::

   The **Logging** tab integrates Grafana Loki. For more information, see the `Loki documentation <https://grafana.com/docs/loki/latest/>`_.

Monitoring
----------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Number of Replicas
     - Displays the count of active replicas in the Deployment. It provides a detailed count for both the total number of replicas and the individual number of API Servers and Runners.
   * - Request Percentile Response Times
     - Displays the response time across different percentiles (for example, P95), giving insight into the range of response speeds that users may experience. It includes the total response time as well as the response time by API endpoints.
   * - Request Volume
     - Tracks the number of requests made to the Deployment over time, which is helpful in understanding the demand and traffic patterns. It includes the total request volume as well as the volume by API endpoints.
   * - Request Success Rate (non-4|5xx responses)
     - Indicates the proportion of requests that are successfully processed without any client-side (4xx) or server-side (5xx) errors. It includes the total success rate as well as the success rate by API endpoints.
   * - CPU Usage
     - Displays the CPU utilization of the Deployment, which includes the total usage as well as the usage by individual components.
   * - Memory Usage
     - Displays the amount of memory being used over time, which includes the total usage as well as the usage by individual components.
   * - GPU Usage
     - Displays the GPU utilization of the Deployment, which includes the total usage as well as the usage by individual components.
   * - GPU Memory Usage
     - Displays the memory usage on the GPU, offering insights into how memory-intensive the GPU tasks are. It includes the total usage as well as the usage by individual components.
   * - GPU Memory Bandwidth Usage
     - Displays the bandwidth usage of the GPU memory, providing data on how intensively the memory is being accessed and at what rate data is being transferred. It includes the total usage as well as the usage by individual components.

Revisions
---------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - ID
     - A unique identifier assigned to each revision of the Deployment.
   * - Deployment Targets
     - The Bento used to create the Deployment.
   * - Creator
     - The user who created this Bento Deployment.
   * - Created at
     - The date and time when the Bento Deployment was initially created.
   * - Operation
     - Actions that can be performed on each revision. For example, you can roll back your Deployment to a specific version.
