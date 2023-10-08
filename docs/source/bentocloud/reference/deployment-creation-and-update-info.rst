==========================================
Deployment creation and update information
==========================================

Deployment creation and update information refers to the set of properties available for creating and updating Bento Deployments.

Configuration types
-------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Type
     - Description
   * - Basic
     - Provides basic configurations of the Deployment, such as Cluster, Endpoint Access Type, and resources for API Server and Runner Pods.
       It is convenient for quickly spinning up a Deployment.
   * - Advanced
     - Provides additional configurations of the Deployment, such as autoscaling behaviors, traffic control, environment variables, and update strategies.
   * - JSON
     - Defines a JSON file to create or update the Deployment, which contains the same fields as those in Advanced. You can download the Deployment information
       in JSON by clicking **Download as JSON**. To create a Deployment from your local machine using the JSON file, run ``bentoml deployment create -f deployment.json``.

Deployment properties
---------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - Cluster
     - The cluster where your Bento Deployment is created.
   * - Kubernetes Namespace
     - The Kubernetes namespace where your Deployment is created.
   * - Deployment Name
     - The name of your Deployment.
   * - Description
     - An introduction to your Deployment, providing additional information on it.
   * - Endpoint Access Type
     - You can manage the endpoint access to your Deployment, choosing between different access levels depending on your needs. This flexibility allows you to control who can access your Deployment, enhancing both security and ease-of-use.

       - **Protected**: The Deployment is accessible to anyone on the internet with a valid User token. This method adds a layer of security by ensuring only those with the token can access the Deployment. It’s suitable when you want your Deployment to be globally accessible but maintain some form of authentication and access control.
       - **Public**: The Deployment is accessible to anyone on the internet without requiring any tokens or credentials. This maximizes access and is suitable when you’re deploying Bentos intended for public use or testing with no sensitive data or operations. Use this option with caution as it does not provide any access control.

   * - Bento Repository
     - Bento repositories act as a centralized hub for managing packaged machine learning models, offering tools for versioning, sharing, retrieval, and deployment. Each Bento repository corresponds to a Bento set, which contains different versions of a specific machine learning model. All the Bento repositories are displayed on the **Bentos** page.
   * - Bento
     - A deployable artifact containing all the application information, such as model files, code, and dependencies. After selecting a Bento repository, you need to specify a Bento version to deploy.
   * - Autoscaling
     - The autoscaling feature dynamically adjusts the number of API Server and Runner Pods within the specified minimum and maximum limits. Min and Max values define the boundaries for scaling, allowing the autoscaler to reduce or increase the number of Pods as needed. This feature supports scaling to zero Pods.
       For Advanced configurations, you can define specific metric thresholds that the autoscaler will use to determine when to adjust the number of Pods. The available metrics for these purposes include:

       - **CPU**: The CPU utilization percentage.
       - **Memory**: The memory utilization.
       - **GPU**: The GPU utilization percentage.
       - **QPS**: The queries per second.

       By setting values for these fields, you are instructing the autoscaler to ensure that the average for each metric does not exceed the specified thresholds. For example, if you set the CPU value to 80, the autoscaler will target an average CPU utilization of 80%.

       Allowed scaling-up behaviors:

       - **Fast (default)**: There is no stabilization window, so the autoscaler can increase the number of Pods immediately if necessary. It can increase the number of Pods by 100% or by 4 Pods, whichever is higher, every 15 seconds.
       - **Stable**: The autoscaler can increase the number of Pods, but it will stabilize the number of Pods for 300 seconds (5 minutes) before deciding to scale up further. It can increase the number of Pods by 100% every 15 seconds.
       - **Disabled**: Scaling-up is turned off.

       Allowed scaling-down behaviors:

       - **Fast**: There is no stabilization window, so the autoscaler can reduce the number of Pods immediately if necessary. It can decrease the number of Pods by 100% or by 4 Pods, whichever is higher, every 15 seconds.
       - **Stable (default)**: The autoscaler can reduce the number of Pods, but it will stabilize the number of Pods for 300 seconds (5 minutes) before deciding to scale down further. It can decrease the number of Pods by 100% every 15 seconds.
       - **Disabled**: Scaling-down is turned off.

   * - Resources per replica
     - You can separately allocate resources for API Servers and Runners using one of the following two strategies:

       - **Basic**: Allows you to select a specific instance type suitable for your workloads. Within both the GPU and CPU groups, there is a set of machine types available for selection. You can customize your selection based on the needs of your application. Every machine type displays information about pricing, the number of GPUs or CPUs, and the memory capacity.
       - **Custom**: Provides greater flexibility, enabling you to specify resource requests and limits for CPU and memory. For Advanced configurations, you are able to set custom resources such as GPU. Min refers to the minimum resource requests allocated to a container, guaranteeing a certain amount of resources at its disposal. Max represents the resource limits for a container, indicating the maximum amount of resources it can use.

   * - Traffic control*
     - You can control the traffic of BentoML API Servers and Runners using the following two ways.

       - **Timeout**: Specify the maximum duration to wait before a response is received. The timeout can be configured in seconds, minutes, and hours. This property is especially useful in managing server load and maintaining responsiveness in high-traffic scenarios. It defaults to 60 seconds.
       - **Request queue**: Set a limit on the maximum number of requests that can be waiting in the processing queue across all API Servers or Runners. By default, there’s no limit, but by setting a specific limit, you can prevent a large backlog of requests from overwhelming your application. If the queue is full, new incoming requests may be rejected until there’s room in the queue. This attribute is useful for managing server loads, especially during periods of high traffic or when dealing with resource-intensive requests.

   * - Environment variables*
     - Environment variables allow you to configure your Bento applications based on the current environment, without the need to hard-code any specific values in your scripts or codebases. They are key-value pairs that can be injected into the Pod where your application runs. You can use them for various purposes like setting up connections to databases, defining paths to dependencies, or any other configuration that your application might need to run.
   * - Deployment strategy*
     - The Deployment strategy determines how traffic is migrated from the old version to the new version of your Bento application.

       - **RollingUpdate**: This strategy incrementally updates Pod replicas to ensure service continuity throughout the update process
       - **Recreate**: All old replicas are deleted before new ones are created. While this ensures a clean transition to the new version, it causes a service interruption during the update, which might not be suitable for all scenarios.
       - **RampedSlowRollout**: This strategy gradually introduces new replicas, adding a new one and then deleting an old one. Although this makes the update speed slower, it minimizes the risk of the update. The service remains available throughout the update, making this strategy ideal for critical applications where downtime is not permissible.
       - **BestEffortControlledRollout**: This strategy strikes a balance between speed and stability. During the update, it maintains a 20% unavailability rate across all replicas to ensure faster update speed. While this strategy allows for quicker transitions, it poses a higher risk as a certain level of downtime during the update has to be accepted. This approach could be suitable when some disruption can be tolerated for the benefit of a faster transition to the new version.

   * - BentoML Configuration*
     - Add additional BentoML configurations to customize the behavior of your Deployment. For more information, see :doc:`/guides/configuration`.

.. note::

   Properties marked with an asterisk (*) are only available for Advanced and JSON configurations.
