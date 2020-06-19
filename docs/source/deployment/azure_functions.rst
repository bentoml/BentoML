Deploying to Azure Functions
============================

Azure Functions is an event driven, compute-on-demand cloud service offered by
Microsoft. Its serverless execution model along with the ability to bring your own
container allow users deploy a BentoML API server with full functionality that scales
base on usage.

In this guide, it will deploy an IrisClassifier BentoService to Azure Functions, make a
prediction request to the deployment endpoint, and then delete the deployment. It will
also show how to start an remote YataiService with Azure credential for deploying to
the Azure Functions.

Prerequisites:
--------------

* An signed in Azure CLI tool on your local machine.

    * Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
    * Sign in Azure CLI: https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli?view=azure-cli-latest

* Docker is installed and running on your local machine

    * Install instruction: https://docs.docker.com/get-docker/

* Python 3.6 or above and required Pypi packages: `bentoml` and `scikit-learn`

    * .. code-block:: bash

            pip install bentoml scikit-learn


Azure Functions deployment with BentoML
---------------------------------------

Run the example project from the :doc:`Quickstart guide <../quickstart>` to create an
IrisClassifier BentoService saved bundle for the deployment:

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    python ./bentoml/guides/quick-start/main.py


Use BentoML CLI tool to get the information of the IrisClassifier BentoService created
above

.. code-block:: bash

    $ bentoml get IrisClassifier:latest

    # Sample output
    {
      "name": "IrisClassifier",
      "version": "20200618124555_310CE0",
      "uri": {
        "type": "LOCAL",
        "uri": "/Users/bozhaoyu/bentoml/repository/IrisClassifier/20200618124555_310CE0"
      },
      "bentoServiceMetadata": {
        "name": "IrisClassifier",
        "version": "20200618124555_310CE0",
        "createdAt": "2020-06-18T19:46:18.675900Z",
        "env": {
          "condaEnv": "name: bentoml-IrisClassifier\nchannels:\n- defaults\ndependencies:\n- python=3.7.3\n- pip\n",
          "pipDependencies": "scikit-learn\npandas\nbentoml==0.8.1",
          "pythonVersion": "3.7.3",
          "dockerBaseImage": "bentoml/model-server:0.8.1"
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
            "inputType": "DataframeInput",
            "docs": "BentoService API",
            "inputConfig": {
              "orient": "records",
              "typ": "frame",
              "is_batch_input": true,
              "input_dtypes": null
            },
            "outputConfig": {
              "cors": "*"
            },
            "outputType": "DefaultOutput",
            "mbMaxLatency": 300,
            "mbMaxBatchSize": 2000
          }
        ]
      }
    }


======================================
Deploy BentoService to Azure Functions
======================================

Deploy to Azure Functions requires a deployment name, BentoService in name:version
format, and a valid Azure location.  You can find list of Azure locations by calling
command `az account list-locations`.

BentoML will create necessary Azure resources, and then build and deploy your
BentoService as docker image to Azure Functions.

.. code-block:: bash

    $ bentoml azure-functions deploy azure-bentoml -b IrisClassifier:20200618124555_310CE0 --location westus

    # sample output
    [2020-06-18 12:54:49,232] INFO - ApplyDeployment (azure-bentoml, namespace dev) succeeded
    Successfully created Azure Functions deployment azure-bentoml
    {
      "namespace": "dev",
      "name": "azure-bentoml",
      "spec": {
        "bentoName": "IrisClassifier",
        "bentoVersion": "20200618124555_310CE0",
        "operator": "AZURE_FUNCTION",
        "azureFunctionOperatorConfig": {
          "location": "westus",
          "premiumPlanSku": "EP1",
          "minInstances": 1,
          "maxBurst": 20,
          "functionAuthLevel": "anonymous"
        }
      },
      "state": {
        "state": "RUNNING",
        "infoJson": {
          "defaultHostName": "dev-azure-bentoml.azurewebsites.net",
          "enabledHostNames": [
            "dev-azure-bentoml.azurewebsites.net",
            "dev-azure-bentoml.scm.azurewebsites.net"
          ],
          "hostNames": [
            "dev-azure-bentoml.azurewebsites.net"
          ],
          "id": "/subscriptions/f01c41a2-72ba-480c-99a6-b3241fada0ac/resourceGroups/dev-azure-bentoml/providers/Microsoft.Web/sites/dev-azure-bentoml",
          "kind": "functionapp,linux,container",
          "lastModifiedTimeUtc": "2020-06-18T19:54:48.540000",
          "location": "West US",
          "name": "dev-azure-bentoml",
          "repositorySiteName": "dev-azure-bentoml",
          "reserved": true,
          "resourceGroup": "dev-azure-bentoml",
          "state": "Running",
          "type": "Microsoft.Web/sites",
          "usageState": "Normal"
        },
        "timestamp": "2020-06-18T19:54:55.456691Z"
      },
      "createdAt": "2020-06-18T19:47:57.385626Z",
      "lastUpdatedAt": "2020-06-18T19:47:57.385659Z"
    }



Use `bentoml azure-functions get` command to retrieve the latest status information of
the deployment.

.. code-block:: bash

    $ bentoml azure-functions get azure-bentoml

    # Sample output
    {
      "namespace": "dev",
      "name": "azure-bentoml",
      "spec": {
        "bentoName": "IrisClassifier",
        "bentoVersion": "20200618124555_310CE0",
        "operator": "AZURE_FUNCTION",
        "azureFunctionOperatorConfig": {
          "location": "westus",
          "premiumPlanSku": "EP1",
          "minInstances": 1,
          "maxBurst": 20,
          "functionAuthLevel": "anonymous"
        }
      },
      "state": {
        "state": "RUNNING",
        "infoJson": {
          "defaultHostName": "dev-azure-bentoml.azurewebsites.net",
          "enabledHostNames": [
            "dev-azure-bentoml.azurewebsites.net",
            "dev-azure-bentoml.scm.azurewebsites.net"
          ],
          "hostNames": [
            "dev-azure-bentoml.azurewebsites.net"
          ],
          "id": "/subscriptions/f01c41a2-72ba-480c-99a6-b3241fada0ac/resourceGroups/dev-azure-bentoml/providers/Microsoft.Web/sites/dev-azure-bentoml",
          "kind": "functionapp,linux,container",
          "lastModifiedTimeUtc": "2020-06-18T19:54:48.540000",
          "location": "West US",
          "name": "dev-azure-bentoml",
          "repositorySiteName": "dev-azure-bentoml",
          "reserved": true,
          "resourceGroup": "dev-azure-bentoml",
          "state": "Running",
          "type": "Microsoft.Web/sites",
          "usageState": "Normal"
        },
        "timestamp": "2020-06-18T19:55:54.292111Z"
      },
      "createdAt": "2020-06-18T19:47:57.385626Z",
      "lastUpdatedAt": "2020-06-18T19:47:57.385659Z"
    }

During Azure Functions initialized stage, it takes a while to download the docker image.
Please wait few minutes before visiting your deployment.

With the default authorization setting, your Azure Functions deployment is visible to
all.  Open your browser and visit the URL in hostNames. You should see the web UI
provided by BentoML API server.

To test the prediction API in the Azure Functions deployment, you could use the web ui
mentioned above or you could make a `curl` request to the endpoint.


.. code-block:: bash

    $  curl -i --request POST --header "Content-Type: application/json" \
        --data '[[5.1, 3.5, 1.4, 0.2]]' \
        "https://dev-azure-bentoml.azurewebsites.net/predict"

    # Sample output

    HTTP/1.1 200 OK
    Content-Length: 3
    Content-Type: application/json
    Server: Kestrel
    Request-Context: appId=cid-v1:c3a39ce6-5bf4-4961-a0de-01c0897b49de
    request_id: 7a75f307-e62b-44f3-b834-7f87f9b96209
    Date: Thu, 18 Jun 2020 20:06:01 GMT

    [0]%


Use `bentoml azure-functions list` to display all active deployments to Azure Functions

.. code-block:: bash

    $ bentoml azure-functions list

    # Sample output
    NAME           NAMESPACE    PLATFORM        BENTO_SERVICE                         STATUS    AGE
    azure-bentoml  dev          azure-function  IrisClassifier:20200618124555_310CE0  running   19 minutes and 22.14 seconds



==========================================
Update existing Azure Functions deployment
==========================================

To update an active Azure Function deployment use `bentoml azure-functions update`
command.

.. code-block: bash

    $ bentoml azure-functions update azure-bentoml -b IrisClassifier:new_version


=================================
Remove Azure Functions deployment
=================================

BentoML will remove all Azure resources created for the deployment.

.. code-block:: bash

    $ bentoml azure-functions delete azure-bentoml


=====================================================================
Deploy and manage Azure Functions deployment with remote YataiService
=====================================================================

BentoML recommends to use remote YataiService for managing and deploying BentoService
when you are working in a team. To deploy Azure Functions in remote YataiService, you
need to provide the Azure credential for it.

In this guide, it will starts a docker container as remote YataiService. After Sign in
with Azure CLI in your local machine, you should be able to find the `accessTokens.json`
in your Azure directory. Now start the BentoML YataiService docker image and mount that
`accessTokens.json` file to the running container.

.. code-block:: bash

    $ docker run -v /Users/bozhaoyu/.azure/accessTokens.json:/home/.azure/accessTokens.json -p 50051:50051 -p 3000:3000 bentoml/yatai-service:latest


After the YataiService docker container is running, in another terminal window, set
yatai service address with `bentoml config set`

.. code-block:: bash

    $ bentoml config set yatai_service.url=127.0.0.1:50051

