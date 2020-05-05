Deploying to AWS Lambda
=======================


AWS Lambda is a great service for quickly deploy service to the cloud for immediate
access. It's ability to auto scale resources base on usage make it attractive to
user who want to save cost and want to scale base on usage without administrative overhead.



Prerequisites
-------------

* An active AWS account configured on the machine with AWS CLI installed and configurated

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
    python ./bentoml/guides/quick-start/main.py

Use `bento list` to get the BentoService name:version tag.


.. code-block:: bash

    > bentoml list

    BentoService                          CREATED_AT                  APIS                        ARTIFACTS
    IrisClassifier:20200121141808_FE78B5  2020-01-21 22:18:25.079723  predict(DataframeHandler)  model(SklearnModelArtifact)


BentoML has great support for AWS Lambda. You can deploy, update and delete Lambda
deployment with single command, and customize deployment to fit your needs with parameters
such as `memory_size` and `timeout`

To deloy BentoService to AWS Lambda, use `bentoml lambda deploy` command.
Deployment name and bento service name:version tag is required.

.. code-block:: bash

    > bentoml lambda deploy my-first-lambda-deployment -b IrisClassifier:20200121141808_FE78B5

    Deploying Lambda deployment -[2020-01-21 14:37:16,838] INFO - Building lambda project
    [2020-01-21 14:38:52,826] INFO - Packaging AWS Lambda project at /private/var/folders/kn/xnc9k74x03567n1mx2tfqnpr0000gn/T/bentoml-temp-qmzs123h ...
    [2020-01-21 14:39:18,834] INFO - Deploying lambda project
    [2020-01-21 14:40:09,265] INFO - ApplyDeployment (my-frist-lambda-deployment, namespace dev) succeeded
    Successfully created AWS Lambda deployment my-frist-lambda-deployment
    {
      "namespace": "dev",
      "name": "my-first-lambda-deployment",
      "spec": {
        "bentoName": "IrisClassifier",
        "bentoVersion": "20200121141808_FE78B5",
        "operator": "AWS_LAMBDA",
        "awsLambdaOperatorConfig": {
          "region": "us-west-2",
          "memorySize": 1024,
          "timeout": 3
        }
      },
      "state": {
        "state": "RUNNING",
        "infoJson": {
          "endpoints": [
            "https://ps6f0sizt8.execute-api.us-west-2.amazonaws.com/Prod/predict"
          ],
          "s3_bucket": "btml-dev-my-first-lambda-deployment-a4a791"
        },
        "timestamp": "2020-01-21T22:40:09.459958Z"
      },
      "createdAt": "2020-01-21T22:37:11.520238Z",
      "lastUpdatedAt": "2020-01-21T22:37:11.520286Z"
    }


BentoML helps you manage the entire process of deploying your BentoService bundle to Lambda.
Verify the deployed resources with AWS CLI tool:

.. code-block:: bash

    > aws cloudformation describe-stacks

    {
        "Stacks": [
            {
                "StackId": "arn:aws:cloudformation:us-west-2:192023623294:stack/dev-my-first-lambda-deployment/dd2a7cf0-3c9e-11ea-8654-02f6ffa9fe66",
                "StackName": "dev-my-first-lambda-deployment",
                "ChangeSetId": "arn:aws:cloudformation:us-west-2:192023623294:changeSet/samcli-deploy1579646359/f9c876ca-ade0-4623-93e9-870ef6e7e1b5",
                "CreationTime": "2020-01-21T22:39:20.156Z",
                "LastUpdatedTime": "2020-01-21T22:39:25.602Z",
                "RollbackConfiguration": {},
                "StackStatus": "CREATE_COMPLETE",
                "DisableRollback": false,
                "NotificationARNs": [],
                "Capabilities": [
                    "CAPABILITY_IAM"
                ],
                "Outputs": [
                    {
                        "OutputKey": "S3Bucket",
                        "OutputValue": "btml-dev-my-first-lambda-deployment-a4a791",
                        "Description": "S3 Bucket for saving artifacts and lambda bundle"
                    },
                    {
                        "OutputKey": "EndpointUrl",
                        "OutputValue": "https://ps6f0sizt8.execute-api.us-west-2.amazonaws.com/Prod",
                        "Description": "URL for endpoint"
                    }
                ],
                "Tags": [],
                "DriftInformation": {
                    "StackDriftStatus": "NOT_CHECKED"
                }
            }
        ]
    }

Tests the deployed service with sample dataset:

.. code-block:: bash

    > curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      https://ps6f0sizt8.execute-api.us-west-2.amazonaws.com/Prod/predict

    HTTP/1.1 200 OK
    Content-Type: application/json
    Content-Length: 3
    Connection: keep-alive
    Date: Tue, 21 Jan 2020 22:43:17 GMT
    x-amzn-RequestId: f49d29ed-c09c-4870-b362-4cf493556cf4
    x-amz-apigw-id: GrC0AEHYPHcF3aA=
    X-Amzn-Trace-Id: Root=1-5e277e7f-e9c0e4c0796bc6f4c36af98c;Sampled=0
    X-Cache: Miss from cloudfront
    Via: 1.1 bb248e7fabd9781d3ed921f068507334.cloudfront.net (CloudFront)
    X-Amz-Cf-Pop: SFO5-C1
    X-Amz-Cf-Id: HZzIJUcEUL8aBI0KcmG35rsG-71KSOcLUNmuYR4wdRb6MZupv9IOpA==

    [0]%

Get the latest state and deployment information such as endpoint and s3 bucket name by
using `bentoml lambda get` command

.. code-block:: bash

    > bentoml lambda get my-first-lambda-deployment

    {
      "namespace": "dev",
      "name": "my-first-lambda-deployment",
      "spec": {
        "bentoName": "IrisClassifier",
        "bentoVersion": "20200121141808_FE78B5",
        "operator": "AWS_LAMBDA",
        "awsLambdaOperatorConfig": {
          "region": "us-west-2",
          "memorySize": 1024,
          "timeout": 3
        }
      },
      "state": {
        "state": "RUNNING",
        "infoJson": {
          "endpoints": [
            "https://ps6f0sizt8.execute-api.us-west-2.amazonaws.com/Prod/predict"
          ],
          "s3_bucket": "btml-dev-my-first-lambda-deployment-a4a791"
        },
        "timestamp": "2020-01-21T22:45:20.861346Z"
      },
      "createdAt": "2020-01-21T22:37:11.520238Z",
      "lastUpdatedAt": "2020-01-21T22:37:11.520286Z"
    }


Use `bentoml lambda list` to have a quick glance of all of the AWS Lambda deployments

.. code-block:: bash

    > bentoml lambda list

    NAME                        NAMESPACE    LABELS    PLATFORM    STATUS    AGE
    my-first-lambda-deployment  dev                    aws-lambda  running   8 minutes and 49.6 seconds



Remove Lambda deployment is also very easy.  Calling `bentoml lambda delete` command will delete the Lambda function and related AWS resources

.. code-block:: bash

    > bentoml lambda delete my-first-lambda-deployment

    Successfully deleted AWS Lambda deployment "my-first-lambda-deployment"
