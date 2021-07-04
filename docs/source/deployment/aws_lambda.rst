Deploying to AWS Lambda
=======================


AWS Lambda is a great service for quickly deploy service to the cloud for immediate
access. It's ability to auto scale resources base on usage make it attractive to
user who want to save cost and want to scale base on usage without administrative overhead.



Prerequisites
-------------

* An active AWS account configured on the machine with AWS CLI installed and configured

  * Install instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html
  * Configure AWS account instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

* AWS SAM CLI tool

  * Install instruction: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html

* Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install


AWS Lambda deployment and management with BentoML
-------------------------------------------------

This guide uses the IrisClassifier BentoService from the :doc:`Quick start guide <../quickstart>`:

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    pip install -r ./bentoml/guides/quick-start/requirements.txt
    python ./bentoml/guides/quick-start/main.py

Use `bentoml list` to get the BentoService name:version tag.


.. code-block:: bash

    > bentoml list

    BentoService                          CREATED_AT                  APIS                        ARTIFACTS
    IrisClassifier:20200121141808_FE78B5  2020-01-21 22:18:25.079723  predict(DataframeInput)  model(SklearnModelArtifact)


BentoML has great support for AWS Lambda. You can deploy, update and delete Lambda
deployment with single command, and customize deployment to fit your needs with parameters
such as `memory_size` and `timeout`

Download and Install BentoML Lambda deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/aws-lambda-deploy
    > cd aws-lambda-deploy
    > pip install -r requirements.txt


Edit the deployment options in `lambda_config.json` file

**Available configuration options for AWS Lambda deployment**
* `region`: AWS region for Lambda deployment
* `timeout`: Timeout per request
* `memory_size`: The memory for your function, set a value between 128 MB and 10,240 MB in 1-MB increments

Create Lambda Deployment

.. code-block:: bash

    > BENTO_BUNDLE_PATH=$(bentoml get IrisClassifier:latest --print-location -q)
    > python deploy.py $BENTO_BUNDLE_PATH my-lambda-deployment lambda_config.json

    # Sample output
    Creating AWS Lambda deployable
    Building SAM template
    Building Image
    0
    Build Succeeded
    Built Artifacts
    ...
    ...
    CloudFormation outputs from deployed stack
    -------------------------------------------------------------------------------------------------
    Outputs
    -------------------------------------------------------------------------------------------------
    Key                 EndpointUrl
    Description         URL for endpoint
    Value               https://j2gm5zn7z9.execute-api.us-west-1.amazonaws.com/Prod
    -------------------------------------------------------------------------------------------------

    Successfully created/updated stack - my-lambda-deployment-stack in us-west-1


Get the latest state and deployment information

.. code-block:: bash

    > python describe.py my-lambda-deployment

    # Sample output
    {
      "StackId": "arn:aws:cloudformation:us-west-1:192023623294:stack/my-lambda-deployment-stack/29c15040-db7a-11eb-a721-028d528946df",
      "StackName": "my-lambda-deployment-stack",
      "StackStatus": "CREATE_COMPLETE",
      "CreationTime": "07/02/2021, 21:12:09",
      "LastUpdatedTime": "07/02/2021, 21:12:20",
      "EndpointUrl": "https://j2gm5zn7z9.execute-api.us-west-1.amazonaws.com/Prod"
    }


Tests the deployed service with sample dataset:

.. code-block:: bash

    > curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      https://j2gm5zn7z9.execute-api.us-west-1.amazonaws.com/Prod/predict

    # Sample output
    HTTP/2 200
    content-type: application/json
    content-length: 3
    date: Sat, 03 Jul 2021 19:14:38 GMT
    x-amzn-requestid: d3b5f156-0859-4f69-8b53-c60e800bc0aa
    x-amz-apigw-id: B6GLLECTSK4FY2w=
    x-amzn-trace-id: Root=1-60e0b714-18a97eb5696cec991c460213;Sampled=0
    x-cache: Miss from cloudfront
    via: 1.1 6af3b573d8970d5db2a4d03354335b85.cloudfront.net (CloudFront)
    x-amz-cf-pop: SEA19-C3
    x-amz-cf-id: ArwZ03gbs6GooNN1fy4mPOgaEpM4h4n9gz2lpLYrHmeXZJuGUJgz0Q==

    [0]%


Removing a Lambda deployment

.. code-block:: bash

    > python delete.py my-lambda-deployment lambda_config.json

    # Sample output
    Delete CloudFormation Stack my-lambda-deployment-stack
    Delete ECR repo my-lambda-deployment-repo



Migrating to BentoML Lambda deployment tool
-------------------------------------------

1. Delete the previous deployment use BentoML CLI tool

.. code-block:: bash

    > bentoml lambda delete DEPLOYMENT_NAME


2. Download and Install BentoML Lambda deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/aws-lambda-deploy
    > cd aws-lambda-deploy
    > pip install -r requirements.txt

3. Deploy to Lambda with deployment tool

.. code-block:: bash

    > BENTO_BUNDLE=$(bentoml get Bento_Name:Bento_version --print-location -q)
    > python deploy.py $BENTO_BUNDLE my_deployment lambda_config.json



.. spelling::

    analytics