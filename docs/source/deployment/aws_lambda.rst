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
    python ./bentoml/guides/quick-start/main.py

Use `bento list` to get the BentoService name:version tag.


.. code-block:: bash

    > bentoml list

    BentoService                          CREATED_AT                  APIS                        ARTIFACTS
    IrisClassifier:20200121141808_FE78B5  2020-01-21 22:18:25.079723  predict(DataframeInput)  model(SklearnModelArtifact)


BentoML has great support for AWS Lambda. You can deploy, update and delete Lambda
deployment with single command, and customize deployment to fit your needs with parameters
such as `memory_size` and `timeout`

To deploy BentoService to AWS Lambda, use `bentoml lambda deploy` command.
Deployment name and bento service name:version tag is required.

.. code-block:: bash

    > bentoml lambda deploy my-first-lambda-deployment -b IrisClassifier:20200121141808_FE78B5

    Deploying Lambda deployment -[2020-01-21 14:37:16,838] INFO - Building lambda project
    [2020-01-21 14:38:52,826] INFO - Packaging AWS Lambda project at /private/var/folders/kn/xnc9k74x03567n1mx2tfqnpr0000gn/T/bentoml-temp-qmzs123h ...
    [2020-01-21 14:39:18,834] INFO - Deploying lambda project
    [2020-01-21 14:40:09,265] INFO - ApplyDeployment (my-first-lambda-deployment, namespace dev) succeeded
    Successfully created AWS Lambda deployment my-first-lambda-deployment
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

If you need to look at the logs of your deployed model, we can view these within AWS CloudWatch. You can get here by searching up `CloudWatch` in your AWS Console. Then, on the left panel, click `Logs > Log Groups` and select your Lambda deployment. The name should be of the form `/aws/lambda/dev-{name}` where `{name}` is the name you used when you deployed it using the CLI. Here, you can look at specific instances of your Lambda function and the logs within it. A typical prediction may look something like the following

.. code-block:: none

    ...
    START RequestId: 11ee8a7a-9884-454a-b008-fd814d9b1781 Version: $LATEST
    [INFO] 2020-06-14T02:13:26.439Z 11ee8a7a-9884-454a-b008-fd814d9b1781 {"event": {"resource": "/predict", "path": "/predict", ...
    END RequestId: 11ee8a7a-9884-454a-b008-fd814d9b1781
    REPORT RequestId: 11ee8a7a-9884-454a-b008-fd814d9b1781 Duration: 14.97 ms Billed Duration: 100 ms Memory Size: 1024 MB...
    ...

If you'd like to have some more detailed analytics into your logs, you may notice that we log some more detailed JSON data as debug info. There are three main fields that are logged. `event` (AWS Lambda Event Object), `prediction` (response body), and `status_code` (HTTP Response Code). You can read more about the `event` object here: https://docs.aws.amazon.com/lambda/latest/dg/services-alb.html. An example of the prediction JSON is as follows,

.. code-block:: bash

    {
        "event": {
            "resource": "/predict",
            "path": "/predict",
            "httpMethod": "POST",
            "headers": {
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "CloudFront-Forwarded-Proto": "https",
                "CloudFront-Is-Desktop-Viewer": "true",
                "CloudFront-Is-Mobile-Viewer": "false",
                "CloudFront-Is-SmartTV-Viewer": "false",
                "CloudFront-Is-Tablet-Viewer": "false",
                "CloudFront-Viewer-Country": "CA",
                "Content-Type": "application/json",
                "Host": "w3y4nf55k0.execute-api.us-east-2.amazonaws.com",
                "Postman-Token": "f785223c-e600-4eea-84a2-8215ebe1afaa",
                "Via": "1.1 98aedae6661e3904540676966998ed89.cloudfront.net (CloudFront)",
                "X-Amz-Cf-Id": "K1cd5UVt__3WEj7DI8kfbi1V5MM4a-v2bRm1Y0kq-mHoOCeCsF_ahg==",
                "X-Amzn-Trace-Id": "Root=1-5ee80803-20ab0d226a290900e7f3d334",
                "X-Forwarded-For": "96.49.202.214, 64.252.141.139",
                "X-Forwarded-Port": "443",
                "X-Forwarded-Proto": "https"
            },
            "multiValueHeaders": {
              ...
            },
            "queryStringParameters": null,
            "multiValueQueryStringParameters": null,
            "pathParameters": null,
            "stageVariables": null,
            "requestContext": {
                "resourceId": "7vnchj",
                "resourcePath": "/predict",
                "httpMethod": "POST",
                "extendedRequestId": "OMYwiHX4iYcF4Zg=",
                "requestTime": "15/Jun/2020:23:45:07 +0000",
                "path": "/Prod/predict",
                "accountId": "558447057402",
                "protocol": "HTTP/1.1",
                "stage": "Prod",
                "domainPrefix": "w3y4nf55k0",
                "requestTimeEpoch": 1592264707383,
                "requestId": "57e19330-67af-4d68-8bb9-4418acb8e880",
                "identity": {
                    "cognitoIdentityPoolId": null,
                    "accountId": null,
                    "cognitoIdentityId": null,
                    "caller": null,
                    "sourceIp": "96.49.202.214",
                    "principalOrgId": null,
                    "accessKey": null,
                    "cognitoAuthenticationType": null,
                    "cognitoAuthenticationProvider": null,
                    "userArn": null,
                    "userAgent": "PostmanRuntime/7.25.0",
                    "user": null
                },
                "domainName": "w3y4nf55k0.execute-api.us-east-2.amazonaws.com",
                "apiId": "w3y4nf55k0"
            },
            "body": "[[5.1, 3.5, 1.4, 0.2]]",
            "isBase64Encoded": false
        },
        "prediction": "[0]",
        "status_code": 200
    }

You can parse this JSON using CloudWatch Logs Insights or ElasticSearch. Within Logs Insights, you can construct a query to visualize the logs that match certain criteria. If, for example, you wanted to view all predictions the returned with a status code of 200, the query would look something like

.. code-block:: none

    fields @timestamp, @message, status_code
    | sort @timestamp desc
    | filter status_code = 200

In this example, `@timestamp` and `@message` represent the time when the log was emitted and the full log message. The third field can be any first level JSON field that were logged (either event info or prediction info).

Removing a Lambda deployment is also very easy.  Calling `bentoml lambda delete` command will delete the Lambda function and related AWS resources

.. code-block:: bash

    > bentoml lambda delete my-first-lambda-deployment

    Successfully deleted AWS Lambda deployment "my-first-lambda-deployment"



=================================================================
Deploy and manage AWS Lambda deployments with remote YataiService
=================================================================

BentoML recommends to use remote YataiService for managing and deploying BentoService
when you are working in a team. To deploy AWS Lambda deployments with remote
YataiService, you need to provide the AWS credentials.

After signed in and configured your AWS CLI in your local machine, you can find the
credentials in your aws directory, `~/.aws/credentials` as key value pairs, with key
name as `aws_access_key_id` and `aws_secret_access_key`

Starts a BentoML YataiService docker image and set the credentials found in
`~/.aws/credentials` as environment variables to the running container.

.. code-block:: bash

    $ docker run -e AWS_ACCESS_KEY_ID=MY-ACCESS-KEY-ID \
        -e AWS_SECRET_ACCESS_KEY=MY_SECRET-ACCESS-KEY \
        -e AWS_DEFAULT_REGION=MY-DEFAULT-REGION \
        -p 50051:50051 -p 3000:3000 bentoml/yatai-service:latest


After the YataiService docker container is running, in another terminal window, set
yatai service address with `bentoml config set`

.. code-block:: bash

    $ bentoml config set yatai_service.url=127.0.0.1:50051


========================================================
Deploy and manage AWS Lambda deployments with Kubernetes
========================================================

Create a Kubernetes secret with the the AWS credentials.

Generate bas64 strings from the AWS credentials from your AWS config file.

.. code-block:: bash

    $ echo $AWS_ACCESS_KEY_ID | base64
    $ echo $AWS_SECRET_KEY | base64
    $ echo $AWS_DEFAULT_REGION | base64


Save the following Kubernetes secret definition into a file name `aws-secret.yaml` and
replace `{access_key_id}`, `{secret_access_key}` and `{default_region}` with the values
generated above,

.. code-block:: yaml

    apiVersion: v1
    kind: Secret
    metadata:
        name: my-aws-secret
    type: Opaque
    data:
        access_key_id: {access_key_id}
        secret_access_key: {secret_access_key}
        default_region: {default_region}


.. code-block:: bash

    $ kubectl apply -f aws-secret.yaml


Confirm the secrete is created successfully by using `kubectl describe` command

.. code-block:: bash

    $kubectl describe secret aws-secret



Copy and paste the code below into a file named `yatai-service.yaml`

.. code-block:: yaml

    apiVersion: v1
    kind: Service
    metadata:
      labels:
        app: yatai-service
      name: yatai-service
    spec:
      ports:
      - name: grpc
        port: 50051
        targetPort: 50051
      - name: web
        port: 3000
        targetPort: 3000
      selector:
        app: yatai-service
      type: LoadBalancer
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      labels:
        app: yatai-service
      name: yatai-service
    spec:
      selector:
        matchLabels:
          app: yatai-service
      template:
        metadata:
          labels:
            app: yatai-service
        spec:
          containers:
          - image: bentoml/yatai-service
            imagePullPolicy: IfNotPresent
            name: yatai-service
            ports:
            - containerPort: 50051
            - containerPort: 3000
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-secret
                  key: access_key_id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-secret
                  key: secret_access_key
            - name: AWS_DEFAULT_REGION
              valueFrom:
                secretKeyRef:
                  name: aws-secret
                  key: default_region


Run `kubectl apply` command to deploy Yatai service to the Kubernetes cluster

.. code-block:: bash

    $ kubectl apply -f yatai-service.yaml



