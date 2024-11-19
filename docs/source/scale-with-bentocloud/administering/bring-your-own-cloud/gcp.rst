===================================
BentoCloud BYOC Setup Guide for GCP
===================================

As part of our commitment to ensuring a smooth and efficient onboarding process, we have prepared this setup guide to help your DevOps team integrate BentoCloud into your GCP account.

Request quotas
--------------

To ensure there are no delays in your cluster setup, please make sure you have enough service quotas in your GCP account prior to starting the setup. If not, please request increased quotas in the project and region where you will deploy BentoCloud.

To request quotas:

1. Visit the `GCP Quotas page <https://console.cloud.google.com/iam-admin/quotas>`_.
2. `Request quotas <https://cloud.google.com/compute/resource-usage#vm_instance_quota>`_ in the correct project and region as per your deployment plan. See the table below for quota details:

   .. list-table::
      :widths: 10 35 25 30
      :header-rows: 1

      * - Type
        - Quota name
        - Required quantity
        - Purpose
      * - CPU
        - ``CPUS``
        - 32
        - Run infrastructure workloads, image builder Pods, and serving instances.
      * - GPU
        - Depending on needs:

          - T4: ``NVIDIA_T4_GPUS``
          - L4: ``NVIDIA_L4_GPUS``
          - A100 40GB: ``NVIDIA_A100_GPUS``
          - A100 80GB: ``NVIDIA_A100_80GB_GPUS``
          - H100 (Committed usage only): ``COMMITTED_NVIDIA_H100_GPUS``

        - Based on needs
        - Run your workloads that require GPUs.

Setup
-----

1. Log in to the `GCP console <https://console.cloud.google.com/>`_ and `create a separate project <https://developers.google.com/workspace/guides/create-project>`_ for BentoCloud to manage resources and permissions efficiently.
2. Go to this newly created project. This can be verified at the top of the console, where the project name is displayed.
3. Access the `API Library <https://console.cloud.google.com/apis/dashboard>`_ and enable the required APIs for BentoCloud to function correctly. You can use `this link <https://console.cloud.google.com/flows/enableapi?apiid=artifactregistry.googleapis.com,autoscaling.googleapis.com,bigquery.googleapis.com,bigquerymigration.googleapis.com,bigquerystorage.googleapis.com,cloudapis.googleapis.com,cloudresourcemanager.googleapis.com,cloudtrace.googleapis.com,compute.googleapis.com,container.googleapis.com,containerfilesystem.googleapis.com,containerregistry.googleapis.com,datastore.googleapis.com,deploymentmanager.googleapis.com,dns.googleapis.com,iam.googleapis.com,iamcredentials.googleapis.com,logging.googleapis.com,monitoring.googleapis.com,networkconnectivity.googleapis.com,oslogin.googleapis.com,pubsub.googleapis.com,redis.googleapis.com,servicemanagement.googleapis.com,serviceusage.googleapis.com,sql-component.googleapis.com,storage-api.googleapis.com,storage-component.googleapis.com,storage.googleapis.com&redirect=https://console.cloud.google.com>`_ for a quick setup.

   .. dropdown:: Required APIs

     .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - API
        - Description
      * - http://artifactregistry.googleapis.com/
        - Artifact Registry API
      * - http://autoscaling.googleapis.com/
        - Cloud Autoscaling API
      * - http://bigquery.googleapis.com/
        - BigQuery API
      * - http://bigquerymigration.googleapis.com/
        - BigQuery Migration API
      * - http://bigquerystorage.googleapis.com/
        - BigQuery Storage API
      * - http://cloudapis.googleapis.com/
        - Google Cloud APIs
      * - http://cloudresourcemanager.googleapis.com/
        - Cloud Resource Manager API
      * - http://cloudtrace.googleapis.com/
        - Cloud Trace API
      * - http://compute.googleapis.com/
        - Compute Engine API
      * - http://container.googleapis.com/
        - Kubernetes Engine API
      * - http://containerfilesystem.googleapis.com/
        - Container File System API
      * - http://containerregistry.googleapis.com/
        - Container Registry API
      * - http://datastore.googleapis.com/
        - Cloud Datastore API
      * - http://deploymentmanager.googleapis.com/
        - Cloud Deployment Manager V2 API
      * - http://dns.googleapis.com/
        - Cloud DNS API
      * - http://iam.googleapis.com/
        - Identity and Access Management (IAM) API
      * - http://iamcredentials.googleapis.com/
        - IAM Service Account Credentials API
      * - http://logging.googleapis.com/
        - Cloud Logging API
      * - http://monitoring.googleapis.com/
        - Cloud Monitoring API
      * - http://networkconnectivity.googleapis.com/
        - Network Connectivity API
      * - http://oslogin.googleapis.com/
        - Cloud OS Login API
      * - http://pubsub.googleapis.com/
        - Cloud Pub/Sub API
      * - http://redis.googleapis.com/
        - Google Cloud Memorystore for Redis API
      * - http://servicemanagement.googleapis.com/
        - Service Management API
      * - http://serviceusage.googleapis.com/
        - Service Usage API
      * - http://sql-component.googleapis.com/
        - Cloud SQL
      * - http://storage-api.googleapis.com/
        - Google Cloud Storage JSON API
      * - http://storage-component.googleapis.com/
        - Cloud Storage
      * - http://storage.googleapis.com/
        - Cloud Storage API

4. Install `the gcloud CLI tool <https://cloud.google.com/sdk/docs/install-sdk>`_ and authenticate your GCP account by running ``gcloud auth login`` in your terminal. Follow the on-screen instructions to complete authentication.
5. Run the setup script, which will set up the necessary infrastructure components for BentoCloud in your GCP project and create a key file.

   a. Before running the script, set your GCP project ID as an environment variable. You can retrieve your project ID by `following the instructions here <https://support.google.com/googleapi/answer/7014113?hl=en>`_.

      .. code-block:: bash

         export PROJECT=<project id>

   b. Run the GCP setup script:

      .. code-block:: bash

         bash <(curl https://l.bentoml.com/bentocloud_gcp_setup_script -sL)

Post setup
----------

Upon completion of the setup script, a service account key file named ``bentocloud-admin-$PROJECT.json`` is created, where ``$PROJECT`` is your GCP project ID. Please send the generated service account key and the GCP region where you want the cluster to be created (e.g. ``us-central1``) to the BentoML team.

.. important::

   For security reasons, it's crucial to transfer this file through a secure channel. Please reach out to your BentoML contact for this step.

Getting help and troubleshooting
--------------------------------

Please reach out to us if you encounter any issues or have questions during the setup process. Our support team is available to assist you with:

- Detailed walkthroughs of each step
- Troubleshooting common issues such as API activation errors, permission issues, or script execution problems
- Best practices for managing BentoCloud in your GCP environment

You can contact our support team at support@bentoml.com or through our support Slack channel.
