Deploying to Azure Functions
============================

Azure Functions is an event driven, compute-on-demand cloud service offered by
Microsoft. Its serverless execution model along with the ability to bring your own
container allow users deploy a BentoML API server with full functionality that scales
base on usage.

In this guide, it will deploy an IrisClassifier BentoService to Azure Functions, make a
prediction request to the deployment endpoint, and then delete the deployment.

Prerequisites:
--------------

* An signed in Azure CLI tool on your local machine.

    * Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
    * Sign in Azure CLI: https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli?view=azure-cli-latest

* Docker is installed and running on your local machine

    * Install instruction: https://docs.docker.com/get-docker/

* Python 3.7 or above and required Pypi packages: `bentoml` and `scikit-learn`

    * .. code-block:: bash

        > pip install bentoml scikit-learn


Azure Functions deployment with BentoML
---------------------------------------

Run the example project from the :doc:`Quickstart guide <../quickstart>` to create an
IrisClassifier BentoService saved bundle for the deployment:

.. code-block:: bash

    > git clone git@github.com:bentoml/BentoML.git
    > pip install -r ./bentoml/guides/quick-start/requirements.txt
    > python ./bentoml/guides/quick-start/main.py


Use BentoML CLI tool to get the information of the IrisClassifier BentoService created
above

.. code-block:: bash

    > bentoml get IrisClassifier:latest

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
            "artifactType": "SklearnModel"
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
              "dtype": null
            },
            "outputConfig": {},
            "outputType": "DefaultOutput",
            "mbMaxLatency": 10000,
            "mbMaxBatchSize": 2000
          }
        ]
      }
    }


======================================
Deploy BentoService to Azure Functions
======================================

Download and Install BentoML Azure Functions deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/azure-functions-deploy
    > cd azure-function-deploy
    > pip install -r requirements.txt

Azure Functions deployment tool creates necessary Azure resources and then build and deploy
BentoBundle as a docker image to Azure Functions

Update the `azure_config.json` file with Azure Functions options that work with the deployment

**Available configuration options for Azure Functions deployment**

* `location`: Azure Function location. Use `az account list-locations` to find list of Azure locations.
* `min_instances`: The number of workers for the deployed app. Default is 1
* `max_burst`: The maximum number of workers for the deployed app Default is 20
* `function_auth_level`: The authentication level for the function. Allowed values: anonymous, function, admin. Default is anonymous. See the link for more information, https://docs.microsoft.com/en-us/java/api/com.microsoft.azure.functions.annotation.httptrigger.authlevel?view=azure-java-stable
* `premium_plan_sku`: The app service plan SKU. Allowed values: EP1, EP2, EP3. Default is EP1. See the link for more information, https://docs.microsoft.com/en-us/azure/azure-functions/functions-premium-plan
* `acr_sku` The SKU for Azure Container Registry. Allowed values: Basic, Classic, Premium, Standard. Default is Standard

.. code-block:: bash

    > BENTO_BUNDLE_PATH=$(bentoml get IrisClassifier:latest --print-location -q)
    > python deploy.py $BENTO_BUNDLE_PATH iris-classifier-demo azure_config.json


Use `describe.py` script to retrieve the latest status information of
the deployment.

.. code-block:: bash

    $ python describe.py iris-classifier-demo

During Azure Functions initialized stage, it takes a while to download the docker image.
Please wait few minutes before visiting your deployment.

With the default authorization setting, your Azure Functions deployment is visible to
all.  Open your browser and visit the URL in hostNames. You should see the web UI
provided by BentoML API server.

To test the prediction API in the Azure Functions deployment, you could use the web UI
mentioned above or you could make a `curl` request to the endpoint.


.. code-block:: bash

    >  curl -i --request POST --header "Content-Type: application/json" \
        --data '[[5.1, 3.5, 1.4, 0.2]]' \
        "https://dev-iris-classifier-demo.azurewebsites.net/predict"

    # Sample output

    HTTP/1.1 200 OK
    Content-Length: 3
    Content-Type: application/json
    Server: Kestrel
    Request-Context: appId=cid-v1:1f23e525-f1cd-471a-ae47-e313f784b99e
    request_id: 525a5c94-41a8-4d9f-9259-0216d3ceb465
    Date: Mon, 22 Jun 2020 21:19:40 GMT

    [0]%


==========================================
Update existing Azure Functions deployment
==========================================

.. code-block:: bash

    > BENTO_BUNDLE_PATH=$(bentoml get IrisClassifier:latest
    > python update.py $BENTO_BUNDLE_PATH iris-classifier-demo azure_config.json


=================================
Remove Azure Functions deployment
=================================

.. code-block:: bash

    > python delete.py iris-classifier-demo


Migrating to BentoML Azure Functions deployment tool
----------------------------------------------------

1. Delete the current deployment use BentoML CLI tool

.. code-block:: bash

    > bentoml azure-functions delete DEPLOYMENT_NAME

2. Download and Install BentoML Azure Functions deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/azure-functions-deploy
    > cd azure-function-deploy
    > pip install -r requirements.txt

3. Deploy to Azure Functions with the deployment tool

.. code-block:: bash

    > BENTO_BUNDLE=$(bentoml get Bento_Name:Bento_version --print-location -q)
    > python deploy.py $BENTO_BUNDLE my_deployment azure_config.json


.. spelling::

    hostNames
