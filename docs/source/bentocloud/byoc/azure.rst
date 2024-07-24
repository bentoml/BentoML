=====================================
BentoCloud BYOC Setup Guide for Azure
=====================================


This document provides step-by-step instructions for configuring the necessary Azure service principal and roles for deploying BentoCloud in your Azure account. By following these steps, you will assign the required roles to a service principal within BentoML's Azure account. This service principal will be used for creating and managing the Azure resources required to operate BentoCloud on your Azure account.

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

   .. note::

      The service principal ID ``d0e2f715-76af-469a-96b9-7d9d9a62b741`` used in the script is the BentoCloud Azure account's service principal.

5. Share the ``account_info.json`` file created by the script with the BentoML team. The file contains non-sensitive information about your Azure account and region needed for the BentoCloud installation.

Getting help and troubleshooting
--------------------------------

Please reach out to us if you encounter any issues or have questions during the setup process. Our support team is available to assist you with:

- Detailed walkthroughs of each step
- Troubleshooting common issues such as API activation errors, permission issues, or script execution problems
- Best practices for managing BentoCloud in your Azure environment

You can contact our support team at support@bentoml.com or through our support Slack channel.
