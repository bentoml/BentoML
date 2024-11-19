==================
Deployment details
==================

Deployment details refer to the properties of a Bento Deployment, such as its metadata, status, monitoring metrics, and revision records.
You set some of these properties when you create the Deployment, and you also have the option to edit some of them as needed after creation.

Playground
----------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Form
     - Interact with the Deployment using a form, which contains the same parameters defined for the Service endpoint.
   * - Python
     - Provide the code to create a Python client to interact with the Deployment.
   * - CURL
     - Provide the CURL command to interact with the Deployment.
   * - Result
     - Display the output by the Deployment.
   * - Authorized/Unauthorized
     - Available only when you enable :ref:`scale-with-bentocloud/deployment/configure-deployments:authorization`. Click it to enter the authorization token.

Replicas
--------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Service Name
     - The name of the BentoML Service.
   * - Instance Type
     - The instance where the Service runs.
   * - Status
     - The status of a Service replica, indicating whether it is running or if it needs your attention. Possible statuses include ``Running``, ``Pending`` and ``Failed``.
   * - Launch Time
     - The time when the Service was deployed.
   * - Replica ID
     - A unique identifier of the Service replica.
   * - Actions
     - Operations that you can perform depending on the role assigned to your account:

       - Logs: Inspect the logs of different containers within the Pod replica. You can use the drop-down menu at the top to switch between containers.
       - Terminal: `Exec into different containers <https://kubernetes.io/docs/tasks/debug/debug-application/get-shell-running-container/>`_ of the Pod replica. You can click **Show File Manager** to upload files to and download files from the container.
       - Debug: `Troubleshoot issues with an ephemeral debug container <https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/#ephemeral-container>`_. You can click **Show File Manager** to upload files to and download files from the container.

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
     - Displays the count of active replicas in the Deployment.
   * - Request Percentile Response Times
     - Displays the response time across different percentiles (for example, P95), giving insight into the range of response speeds that users may experience. It includes the total response time as well as the response time by API endpoints.
   * - Request Volume
     - Tracks the number of requests made to the Deployment over time, which is helpful in understanding the demand and traffic patterns. It includes the total request volume as well as the volume by API endpoints.
   * - Request Success Rate (non-4|5xx responses)
     - Indicates the proportion of requests that are successfully processed without any client-side (4xx) or server-side (5xx) errors. It includes the total success rate as well as the success rate by API endpoints.
   * - In-Progress Request Volume
     - Tracks the number of requests currently being processed but have not yet finished. This metric is important for understanding the real-time load on the server and helps you identify bottlenecks or potential performance issues.
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
   * - Revision ID
     - A unique identifier assigned to each revision of the Deployment.
   * - Bento
     - The Bento used to create the Deployment.
   * - Created by
     - The user who created this Bento Deployment.
   * - Created at
     - The date and time when the Bento Deployment was initially created.
   * - Operations
     - Actions that can be performed on each revision. For example, you can roll back your Deployment to a specific version.
