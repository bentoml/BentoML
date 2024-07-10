=====================================
BentoCloud BYOC Setup Guide for Azure
=====================================

As part of our commitment to ensuring a smooth and efficient onboarding process, we have prepared this setup guide to help your DevOps team integrate BentoCloud into your Azure account.

Setup
-----

1. Log in to `the Azure Portal <https://www.notion.so/BentoCloud-BYOC-Setup-Guide-for-Azure-b2e4a377ba6549d2a9d16967c65d9591?pvs=21>`_.
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

Post setup
----------

Upon completion of the setup script, a file named ``bcAdminSP.json`` will be created, containing the credentials and subscription ID to manage Azure resources under the specified subscription. Please send the file and the Azure region to the BentoML team.

.. important::
    
    For security reasons, it's crucial to transfer this file through a secure channel. Please reach out to your BentoML contact for this step.

Getting help and troubleshooting
--------------------------------

Please reach out to us if you encounter any issues or have questions during the setup process. Our support team is available to assist you with:

- Detailed walkthroughs of each step
- Troubleshooting common issues such as API activation errors, permission issues, or script execution problems
- Best practices for managing BentoCloud in your Azure environment

You can contact our support team at support@bentoml.com or through our support Slack channel.