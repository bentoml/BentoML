=====================================
BentoCloud BYOC Setup Guide for Azure
=====================================


This document provides step-by-step instructions for configuring the necessary Azure service principal and roles for deploying BentoCloud in your Azure account. By following these steps, you will assign the required roles to a service principal within BentoML's Azure account. This service principal will be used for creating and managing the Azure resources required to operate BentoCloud on your Azure account.

Request quotas
--------------

To ensure there are no delays in your cluster setup, please make sure you have enough service quotas in your Azure account prior to starting the setup. If not, please request increased quotas in the subscription and region where you will deploy BentoCloud.

To request quotas:

1. Sign in to `the Azure portal <https://portal.azure.com/>`_ and enter ``Quotas`` into the search box, then select **Quotas**. On the **Overview** page, select **Compute**.
2. `Request quotas <https://cloud.google.com/compute/resource-usage#vm_instance_quota>`_ in the correct subscription and region as per your deployment plan. See the table below for quota details:

   .. list-table::
      :widths: 10 35 25 30
      :header-rows: 1

      * - Type
        - Quota name
        - Required quantity
        - Purpose
      * - CPU
        - ``Total Regional vCPUs`` and ``Standard DSv3 Family vCPUs``
        - 32
        - Run infrastructure workloads, image builder Pods, and serving instances.
      * - GPU
        - Depending on needs:

          - T4: ``Standard NCASv3_T4 Family vCPUs``
          - A100: ``Standard NCADS_A100_v4 Family vCPUs``

        - Based on needs
        - Run your workloads that require GPUs.

Setup
-----

1. Log in to `the Azure Portal <https://azure.microsoft.com/en-us/get-started/azure-portal>`_.
2. Install `the Azure CLI <https://learn.microsoft.com/en-us/cli/azure/install-azure-cli>`_ and log in to your Azure account by running ``az login``.
3. If you have multiple subscriptions, set the desired one by running:

   .. code-block:: bash

      az account set --subscription <Subscription Name or ID>

4. Run the setup script, which will configure necessary Azure resources such as virtual machines, networks, and other services required for BentoCloud. You will be prompted to specify a region for BentoCloud setup.

   .. code-block:: bash

      bash <(curl https://l.bentoml.com/bentocloud_azure_setup_script -sL)

   .. note::

      The script uses ``jq`` for parsing JSON output from Azure CLI commands. Ensure ``jq`` is installed on your system.

   The permission set in the script allows for the creation and management of all required resources, including Azure Kubernetes Service, Blob Storage, and Redis Cache, for the setup and maintenance of BentoCloud cluster on Azure. The scope of the grant is strictly limited to the resource group ``bentocloud-<region>`` and does not grant permissions to any other resources.

5. Share the ``bcAdminSP.json`` file created by the script with the BentoML team through a secure channel.

Getting help and troubleshooting
--------------------------------

Please reach out to us if you encounter any issues or have questions during the setup process. Our support team is available to assist you with:

- Detailed walkthroughs of each step
- Troubleshooting common issues such as API activation errors, permission issues, or script execution problems
- Best practices for managing BentoCloud in your Azure environment

You can contact our support team at support@bentoml.com or through our support Slack channel.
