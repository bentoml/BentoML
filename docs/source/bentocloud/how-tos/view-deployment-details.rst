=======================
View Deployment details
=======================

This page describes how to view the detailed information of a Deployment on the BentoCloud console.

1. In the navigation sidebar, click **Deployments**.
2. The **Deployments** page lists all Deployments in your BentoCloud account. Select the Deployment you want to view. The page opens with the **Overview** tab selected.
3. The Deployment details page provides a comprehensive view of a specific Deployment, with several tabs that allow you to inspect various aspects of the Deployment.

   * **Overview**: Provides information about the Deployment's general health and settings, including its current status, the URL endpoint for the deployed Bento, resource allocations, and basic configuration details.
   * **Replicas**: Provides detailed information about the individual replicas of the Deployment. It displays each replica's status, type, the cluster and node it's running on, the time it started, and any operations that can be performed on them. This tab is useful for understanding the scale and distribution of the Deployment.
   * **Logging**: Contains a live feed of logs related to the Deployment. It allows you to search for logs, troubleshoot issues, and analyze the behavior of the application through the logs of different components.
   * **Monitoring**: Features metrics and graphs that track the performance of the Deployment over time, such as CPU, memory, and GPU utilization. They are are crucial for performance tuning and ensuring that the Deployment is running efficiently and has sufficient resources.
   * **Revisions**: Maintains a historical record of Deployment revisions over time. Each entry represents a different version of the Deployment, showing who made the change as well as when it was made, and providing options to roll back to previous versions if necessary. This is particularly useful for maintaining a stable environment and for audit purposes.

To learn more about each property on the Deployment details page, see :doc:`/bentocloud/reference/deployment-details`.
