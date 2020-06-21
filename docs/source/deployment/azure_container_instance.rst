
Deploying to Azure Container Instance
=====================================

Microsoft Azure container instances is a service for running Docker container without
managing servers. It is a great option for running BentoService that requires a lot of
computing resources.

This guide demonstrates how to deploy a scikit-learn based iris classifier model with
BentoML to Azure Container Instances. The same deployment steps are also applicable for models
trained with other machine learning frameworks, see more BentoML examples :doc:`here <../examples>`.


Prerequisites
-------------

* Azure CLI tool

  * install instruction: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

* Docker is installed and running on your machine.

  * Install instruction: https://docs.docker.com/install

* Python 3.6 or above and required packages: `bentoml` and `scikit-learn`:

  * .. code-block:: bash

        pip install bentoml scikit-learn


Deploying BentoService to Azure Container Instance
--------------------------------------------------

Run the example project from the :doc:`quick start guide <../quickstart>` to create the
BentoML saved bundle for deployment:


.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    python ./bentoml/guides/quick-start/main.py

Verify the saved bundle created:

.. code-block:: bash

    $ bentoml get IrisClassifier:latest

    # sample output
    {
      "name": "IrisClassifier",
      "version": "20200121141808_FE78B5",
      "uri": {
        "type": "LOCAL",
        "uri": "/Users/bozhaoyu/bentoml/repository/IrisClassifier/20200121141808_FE78B5"
      },
      "bentoServiceMetadata": {
        "name": "IrisClassifier",
        "version": "20200121141808_FE78B5",
        "createdAt": "2020-01-21T22:18:25.079723Z",
        "env": {
          "condaEnv": "name: bentoml-IrisClassifier\nchannels:\n- defaults\ndependencies:\n- python=3.7.3\n- pip\n",
          "pipDependencies": "bentoml==0.5.8\nscikit-learn",
          "pythonVersion": "3.7.3"
        },
        "artifacts": [
          {
            "name": "model",
            "artifactType": "SklearnModelArtifact"
          }
        ],
        "apis": [
          {
            "name": "predict",
            "InputType": "DataframeInput",
            "docs": "BentoService API"
          }
        ]
      }
    }


The BentoML saved bundle created can now be used to start a REST API Server hosting the
BentoService and available for sending test request:

.. code-block:: bash

    # Start BentoML API server:
    bentoml serve IrisClassifier:latest


.. code-block:: bash

    # Send test request:
    curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      http://localhost:5000/predict


===================
Configure Azure CLI
===================

.. code-block:: bash

    $ az login

    # Sample output

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

    $ az group create --name iris-classifier --location eastus

    # Sample output
    {
      "id": "/subscriptions/f01c41a2-72ba-480c-99a6-b3241fada0ac/resourceGroups/iris-classifier",
      "location": "eastus",
      "managedBy": null,
      "name": "iris-classifier",
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

    $ az acr create --resource-group iris-classifier --name bentomlirisclassifier --sku Basic --admin-enabled true

    # Sample output

    {- Finished ..
      "adminUserEnabled": true,
      "creationDate": "2020-04-21T04:49:36.301601+00:00",
      "dataEndpointEnabled": false,
      "dataEndpointHostNames": [],
      "encryption": {
        "keyVaultProperties": null,
        "status": "disabled"
      },
      "id": "/subscriptions/f01c41a2-72ba-480c-99a6-b3241fada0ac/resourceGroups/iris-classifier/providers/Microsoft.ContainerRegistry/registries/bentomlirisclassifier",
      "identity": null,
      "location": "eastus",
      "loginServer": "bentomlirisclassifier.azurecr.io",
      "name": "bentomlirisclassifier",
      "networkRuleSet": null,
      "policies": {
        "quarantinePolicy": {
          "status": "disabled"
        },
        "retentionPolicy": {
          "days": 7,
          "lastUpdatedTime": "2020-04-21T04:49:37.160402+00:00",
          "status": "disabled"
        },
        "trustPolicy": {
          "status": "disabled",
          "type": "Notary"
        }
      },
      "privateEndpointConnections": [],
      "provisioningState": "Succeeded",
      "resourceGroup": "iris-classifier",
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

    $ az acr login --name bentomlirisclassifier

    Login Succeeded


.. code-block:: bash

    $ az acr show --name BentoMLIrisClassifier --query loginServer --output table

    # Sample output

    Result
    --------------------------------
    bentomlirisclassifier.azurecr.io


==================================
Build and push docker image to ACR
==================================

.. code-block:: bash

    # Install jq, the command-line JSON processor: https://stedolan.github.io/jq/download/
    $ saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")
    $ docker build -t bentomlirisclassifier.azurecr.io/iris-classifier $saved_path

    # Sample output

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
    Removing intermediate container bb4fd6e496e2
    ---> 264cff2cb98e
    Step 14/15 : ENV FLAGS=""
    ---> Running in f2f0e8b74e01
    Removing intermediate container f2f0e8b74e01
    ---> 4a75521e1a9d
    Step 15/15 : CMD ["bentoml serve-gunicorn /bento $FLAGS"]
    ---> Running in 5ebd6bb79077
    Removing intermediate container 5ebd6bb79077
    ---> 0cb0ac545be1
    Successfully built 0cb0ac545be1
    Successfully tagged bentomlirisclassifier.azurecr.io/iris-classifier:latest


.. code-block:: bash

    $ docker push bentomlirisclassifier.azurecr.io/iris-classifier

    # Sample output

    The push refers to repository [bentomlirisclassifier.azurecr.io/iris-classifier]
    ...
    latest: digest: sha256:4b747c7d4db55278feb20caac6a5cf0ca74fae998b808d5cf2e5a20b3cde4303 size: 2227

=========================================================
Deploying docker image in ACR as Azure container instance
=========================================================

Retrieve registry username and password for container deployment

.. code-block:: bash

    $ az acr repository list --name bentomlirisclassifier --output table

    # Sample output

    Result
    ---------------
    iris-classifier


.. code-block:: bash

    $ az acr credential show -n bentomlirisclassifier

    # Sample output

    {
      "passwords": [
        {
          "name": "password",
          "value": "i/qE2Eu/Ngv344HjfOEPjNKkN9hHre+k"
        },
        {
          "name": "password2",
          "value": "NIoodtFcfhI3YtReyUnCiT=ChOL8ef+X"
        }
      ],
      "username": "bentomlirisclassifier"
    }

Deploying image as Azure container. `registry-username` and `registry-password` are from previous command's output

.. code-block:: bash

    $ az container create --resource-group iris-classifier \
        --name bentomlirisclassifier \
        --image bentomlirisclassifier.azurecr.io/iris-classifier \
        --cpu 1 \
        --memory 1 \
        --registry-login-server bentomlirisclassifier.azurecr.io \
        --registry-username bentomlirisclassifier \
        --registry-password i/qE2Eu/Ngv344HjfOEPjNKkN9hHre+k \
        --dns-name-label bentomlirisclassifier777 \
        --ports 5000

    # Sample output

    {- Finished ..
      "containers": [
        {
          "command": null,
          "environmentVariables": [],
          "image": "bentomlirisclassifier.azurecr.io/iris-classifier",
          "instanceView": {
            "currentState": {
              "detailStatus": "",
              "exitCode": null,
              "finishTime": null,
              "startTime": "2020-04-21T05:15:57+00:00",
              "state": "Running"
            },
            "events": [
              {
                "count": 1,
                "firstTimestamp": "2020-04-21T05:12:55+00:00",
                "lastTimestamp": "2020-04-21T05:12:55+00:00",
                "message": "pulling image \"bentomlirisclassifier.azurecr.io/iris-classifier\"",
                "name": "Pulling",
                "type": "Normal"
              },
              {
                "count": 1,
                "firstTimestamp": "2020-04-21T05:15:54+00:00",
                "lastTimestamp": "2020-04-21T05:15:54+00:00",
                "message": "Successfully pulled image \"bentomlirisclassifier.azurecr.io/iris-classifier\"",
                "name": "Pulled",
                "type": "Normal"
              },
              {
                "count": 1,
                "firstTimestamp": "2020-04-21T05:15:56+00:00",
                "lastTimestamp": "2020-04-21T05:15:56+00:00",
                "message": "Created container",
                "name": "Created",
                "type": "Normal"
              },
              {
                "count": 1,
                "firstTimestamp": "2020-04-21T05:15:57+00:00",
                "lastTimestamp": "2020-04-21T05:15:57+00:00",
                "message": "Started container",
                "name": "Started",
                "type": "Normal"
              }
            ],
            "previousState": null,
            "restartCount": 0
          },
          "livenessProbe": null,
          "name": "bentomlirisclassifier",
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
      "id": "/subscriptions/f01c41a2-72ba-480c-99a6-b3241fada0ac/resourceGroups/iris-classifier/providers/Microsoft.ContainerInstance/containerGroups/bentomlirisclassifier",
      "identity": null,
      "imageRegistryCredentials": [
        {
          "password": null,
          "server": "bentomlirisclassifier.azurecr.io",
          "username": "bentomlirisclassifier"
        }
      ],
      "instanceView": {
        "events": [],
        "state": "Running"
      },
      "ipAddress": {
        "dnsNameLabel": "bentomlirisclassifier777",
        "fqdn": "bentomlirisclassifier777.eastus.azurecontainer.io",
        "ip": "20.185.15.187",
        "ports": [
          {
            "port": 5000,
            "protocol": "TCP"
          }
        ],
        "type": "Public"
      },
      "location": "eastus",
      "name": "bentomlirisclassifier",
      "networkProfile": null,
      "osType": "Linux",
      "provisioningState": "Succeeded",
      "resourceGroup": "iris-classifier",
      "restartPolicy": "Always",
      "tags": {},
      "type": "Microsoft.ContainerInstance/containerGroups",
      "volumes": null
    }

Use `az container show` command to fetch container instance state

.. code-block:: bash

    $ az container show --resource-group iris-classifier --name bentomlirisclassifier --query instanceView.state

    "Running"


We can use the same `az container show` command to retrieve endpoint address

.. code-block:: bash

    $ az container show --resource-group iris-classifier --name bentomlirisclassifier --query ipAddress.fqdn

    "bentomlirisclassifier777.eastus.azurecontainer.io"


===============================================================
Validate Azure container instance with sample data POST request
===============================================================

.. code-block:: bash

    $ curl -X \
        POST "http://bentomlirisclassifier777.eastus.azurecontainer.io:5000/predict" \
        --header "Content-Type: application/json" \
        -d '[[5.1, 3.5, 1.4, 0.2]]'

    [0]


=================================
Clean up Azure container instance
=================================

.. code-block:: bash

    az group delete --name sentiment_azure
