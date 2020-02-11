
Deploying to Azure Container Instance
=====================================

Azure container instances allows you to run docker containers easily on Azure without managing servers.



Prerequisites
-------------

1. Azure CLI tool
* install instruction: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

2. Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install


Deploying BentoService to Azure Container Instance
--------------------------------------------------

===================
Configure Azure CLI
===================

.. code-block:: bash

    > az login

    You have logged in. Now let us find all the subscriptions to which you have access...
    [
      {
        "cloudName": "AzureCloud",
        "id": "d3fe34fd-019d-47b0-a485-de3688e03bdd",
        "isDefault": true,
        "name": "Azure subscription 1",
        "state": "Enabled",
        "tenantId": "1f81e1a8-b059-4e1f-ab49-3ec3c0547d92",
        "user": {
          "name": "7lagrange@gmail.com",
          "type": "user"
        }
      }
    ]


.. code-block:: bash

    > az group create --name sentiment_azure --location eastus

    {
      "id": "/subscriptions/d3fe34fd-019d-47b0-a485-de3688e03bdd/resourceGroups/sentiment_azure",
      "location": "eastus",
      "managedBy": null,
      "name": "sentiment_azure",
      "properties": {
        "provisioningState": "Succeeded"
      },
      "tags": null,
      "type": "Microsoft.Resources/resourceGroups"
    }

=========================================================
Create and configure Azure ACR (Azure Container Registry)
=========================================================

.. code-block:: bash

    > az acr create --resource-group sentiment_azure --name bentosentimentlrmodel --sku Basic --admin-enabled true

    {
      "adminUserEnabled": true,
      "creationDate": "2020-01-20T22:04:09.741079+00:00",
      "id": "/subscriptions/d3fe34fd-019d-47b0-a485-de3688e03bdd/resourceGroups/sentiment_azure/providers/Microsoft.ContainerRegistry/registries/bentosentimentlrmodel",
      "location": "eastus",
      "loginServer": "bentosentimentlrmodel.azurecr.io",
      "name": "bentosentimentlrmodel",
      "networkRuleSet": null,
      "policies": {
        "quarantinePolicy": {
          "status": "disabled"
        },
        "retentionPolicy": {
          "days": 7,
          "lastUpdatedTime": "2020-01-20T22:04:11.344403+00:00",
          "status": "disabled"
        },
        "trustPolicy": {
          "status": "disabled",
          "type": "Notary"
        }
      },
      "provisioningState": "Succeeded",
      "resourceGroup": "sentiment_azure",
      "sku": {
        "name": "Basic",
        "tier": "Basic"
      },
      "status": null,
      "storageAccount": null,
      "tags": {},
      "type": "Microsoft.ContainerRegistry/registries"
    }


.. code-block:: bash

    > az acr login --name bentosentimentlrmodel

    Login Succeeded


.. code-block:: bash

    > az acr show --name BentoSentimentLRModel --query loginServer --output table

    Result
    --------------------------------
    bentosentimentlrmodel.azurecr.io


==================================
Build and push docker image to ACR
==================================

.. code-block:: bash

    > cd '/Users/hongjian/bentoml/repository/SentimentLRModel/20200120135559_A351E9'
    > docker build -t bentosentimentlrmodel.azurecr.io/sentimentlrmodel .

    Sending build context to Docker daemon  8.314MB
    Step 1/12 : FROM continuumio/miniconda3:4.7.12
    ---> 406f2b43ea59
    Step 2/12 : ENTRYPOINT [ "/bin/bash", "-c" ]
    ---> Using cache
    ---> 26c44e044c6f
    Step 3/12 : EXPOSE 5000
    ---> Using cache
    ---> 876689dac8b2
    ...
    ...
    ...
    Step 12/12 : CMD ["bentoml serve-gunicorn /bento"]
    ---> Running in 8e1ef8bfb06f
    Removing intermediate container 8e1ef8bfb06f
    ---> f0b2577e7b09
    Successfully built f0b2577e7b09
    Successfully tagged bentosentimentlrmodel.azurecr.io/sentimentlrmodel:latest


.. code-block:: bash

    > docker push bentosentimentlrmodel.azurecr.io/sentimentlrmodel

    The push refers to repository [bentosentimentlrmodel.azurecr.io/sentimentlrmodel]

    4358219f: Preparing
    6e8a3988: Preparing
    25e8c992: Preparing
    68afe3bd: Preparing
    1e1a7808: Preparing
    cb249b79: Preparing
    190fd43a: Preparing
    4358219f: Pushing  183.1MB/200.3MB7APushing  43.19MBPushing  66.68MB/150.5MBPushing  72.36MB/160.8MB68afe3bd: Pushing    404MB/1.109GB68afe3bd: Pushing  1.109GB/1.109GBPushing    526MB/1.109GB68afe3bd: Pushed   1.115GBlatest: digest: sha256:8a680917935dd096c296147b722c2f2002b7e5c8c2a382db2161e9c64a752c74 size: 2012

===================================================
Deploying docker in ACR as Azure container instance
===================================================

Retrieve registry username and password for container deployment

.. code-block:: bash

    > az acr repository list --name BentoSentimentLRModel --output table

    Result
    ----------------
    sentimentlrmodel


.. code-block:: bash

    > az acr credential show -n BentoSentimentLRModel

    {
      "passwords": [
        {
          "name": "password",
          "value": "+dqLfyU44bJmJTBxXckeDvanxDDTrcCU"
        },
        {
          "name": "password2",
          "value": "KZ7qsX5gvleMQT5jZ=BSoh+jam8l+nAO"
        }
      ],
      "username": "bentosentimentlrmodel"
    }

Deploying image as Azure container. `registry-username` and `registry-password` are from previous command's output

.. code-block:: bash

    > az container create --resource-group sentiment_azure \
      --name sentimentlrmodel \
      --image bentosentimentlrmodel.azurecr.io/sentimentlrmodel \
      --cpu 1 \
      --memory 1 \
      --registry-login-server bentosentimentlrmodel.azurecr.io \
      --registry-username bentosentimentlrmodel \
      --registry-password KZ7qsX5gvleMQT5jZ=BSoh+jam8l+nAO \
      --dns-name-label bentosentimentlrmodel777 \
      --ports 5000

    {- Finished ..
      "containers": [
        {
          "command": null,
          "environmentVariables": [],
          "image": "bentosentimentlrmodel.azurecr.io/sentimentlrmodel",
          "instanceView": {
            "currentState": {
              "detailStatus": "",
              "exitCode": null,
              "finishTime": null,
              "startTime": "2020-01-20T22:43:38+00:00",
              "state": "Running"
            },
            "events": [
              {
                "count": 1,
                "firstTimestamp": "2020-01-20T22:41:32+00:00",
                "lastTimestamp": "2020-01-20T22:41:32+00:00",
                "message": "pulling image \"bentosentimentlrmodel.azurecr.io/sentimentlrmodel\"",
                "name": "Pulling",
                "type": "Normal"
              },
              {
                "count": 1,
                "firstTimestamp": "2020-01-20T22:43:35+00:00",
                "lastTimestamp": "2020-01-20T22:43:35+00:00",
                "message": "Successfully pulled image \"bentosentimentlrmodel.azurecr.io/sentimentlrmodel\"",
                "name": "Pulled",
                "type": "Normal"
              },
              {
                "count": 1,
                "firstTimestamp": "2020-01-20T22:43:38+00:00",
                "lastTimestamp": "2020-01-20T22:43:38+00:00",
                "message": "Created container",
                "name": "Created",
                "type": "Normal"
              },
              {
                "count": 1,
                "firstTimestamp": "2020-01-20T22:43:38+00:00",
                "lastTimestamp": "2020-01-20T22:43:38+00:00",
                "message": "Started container",
                "name": "Started",
                "type": "Normal"
              }
            ],
            "previousState": null,
            "restartCount": 0
          },
          "livenessProbe": null,
          "name": "sentimentlrmodel",
          "ports": [
            {
              "port": 5000,
              "protocol": "TCP"
            }
          ],
          "readinessProbe": null,
          "resources": {
            "limits": null,
            "requests": {
              "cpu": 1.0,
              "gpu": null,
              "memoryInGb": 1.0
            }
          },
          "volumeMounts": null
        }
      ],
      "diagnostics": null,
      "dnsConfig": null,
      "id": "/subscriptions/d3fe34fd-019d-47b0-a485-de3688e03bdd/resourceGroups/sentiment_azure/providers/Microsoft.ContainerInstance/containerGroups/sentimentlrmodel",
      "identity": null,
      "imageRegistryCredentials": [
        {
          "password": null,
          "server": "bentosentimentlrmodel.azurecr.io",
          "username": "bentosentimentlrmodel"
        }
      ],
      "instanceView": {
        "events": [],
        "state": "Running"
      },
      "ipAddress": {
        "dnsNameLabel": "bentosentimentlrmodel777",
        "fqdn": "bentosentimentlrmodel777.eastus.azurecontainer.io",
        "ip": "52.152.188.36",
        "ports": [
          {
            "port": 5000,
            "protocol": "TCP"
          }
        ],
        "type": "Public"
      },
      "location": "eastus",
      "name": "sentimentlrmodel",
      "networkProfile": null,
      "osType": "Linux",
      "provisioningState": "Succeeded",
      "resourceGroup": "sentiment_azure",
      "restartPolicy": "Always",
      "tags": {},
      "type": "Microsoft.ContainerInstance/containerGroups",
      "volumes": null
    }

Use `az container show` command to fetch container instace state

.. code-block:: bash

    > az container show --resource-group sentiment_azure --name sentimentlrmodel --query instanceView.state

    "Running"


We can use the same `az container show` command to retreive endpoint address

.. code-block:: bash

    > az container show --resource-group sentiment_azure --name sentimentlrmodel --query ipAddress.fqdn

    "bentosentimentlrmodel777.eastus.azurecontainer.io"


===============================================================
Validate Azure container instance with sample data POST request
===============================================================

.. code-block:: bash

    > curl -X \
      POST "http://bentosentimentlrmodel777.eastus.azurecontainer.io:5000/predict" \
      --header "Content-Type: application/json" \
      -d '["good movie", "bad food", "i feel happy today"]'

    [4, 0, 4]


=================================
Clean up Azure container instance
=================================

.. code-block:: bash

    > az group delete --name sentiment_azure