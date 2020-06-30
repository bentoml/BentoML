Deploying to AWS SageMaker
==========================

AWS Sagemaker is a fully managed services for quickly building ML models. BentoML provides great support
for deploying BentoService to AWS Sagemaker without additional process and work from user. With BentoML,
users can enjoy the great system performance from Sagemaker with any popular ML frameworks.


Prerequisites
------------

* An active AWS account configured on the machine with AWS CLI installed and configured

  * Install instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html
  * Configure AWS account instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

* Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install



AWS Sagemaker deployment and management with BentoML
----------------------------------------------------

This guide uses the IrisClassifier BentoService from the :doc:`Quick start guide <../quickstart>`:

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    python ./bentoml/guides/quick-start/main.py


Use `bento list` to get the BentoService name:version tag.

.. code-block:: bash

    > bentoml list

    BentoService                          CREATED_AT                  APIS                        ARTIFACTS
    IrisClassifier:20200121141808_FE78B5  2020-01-21 22:18:25.079723  predict(DataframeInput)  model(SklearnModelArtifact)


Deploy to Sagemaker requires a deployment name, BentoService name:version tag, and api name from the
BentoService bundle. We apply those values to  `bentoml sagemaker deploy`.


.. code-block:: bash


    > bentoml sagemaker deploy my-first-sagemaker-deployment -b IrisClassifier:20200121141808_FE78B5 --api-name predict

    Deploying Sagemaker deployment \[2020-01-21 15:26:43,548] INFO - Step 1/11 : FROM continuumio/miniconda3:4.7.12
    ...
    ...
    [2020-01-21 15:27:49,201] INFO - Successfully built d72c7deafa31
    [2020-01-21 15:27:49,212] INFO - Successfully tagged 192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-sagemaker:20200121141808_FE78B5
    [2020-01-21 15:29:31,814] INFO - ApplyDeployment (my-first-sagemaker-deployment, namespace dev) succeeded
    Successfully created AWS Sagemaker deployment my-first-sagemaker-deployment
    {
      "namespace": "dev",
      "name": "my-first-sagemaker-deployment",
      "spec": {
        "bentoName": "IrisClassifier",
        "bentoVersion": "20200121141808_FE78B5",
        "operator": "AWS_SAGEMAKER",
        "sagemakerOperatorConfig": {
          "region": "us-west-2",
          "instanceType": "ml.m4.xlarge",
          "instanceCount": 1,
          "apiName": "predict"
        }
      },
      "state": {
        "state": "RUNNING",
        "infoJson": {
          "EndpointName": "dev-my-first-sagemaker-deployment",
          "EndpointArn": "arn:aws:sagemaker:us-west-2:192023623294:endpoint/dev-my-first-sagemaker-deployment",
          "EndpointConfigName": "dev-my-first-sag-IrisClassifier-20200121141808-FE78B5",
          "ProductionVariants": [
            {
              "VariantName": "dev-my-first-sag-IrisClassifier-20200121141808-FE78B5",
              "DeployedImages": [
                {
                  "SpecifiedImage": "192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-sagemaker:20200121141808_FE78B5",
                  "ResolvedImage": "192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-sagemaker@sha256:cd723a363bcbad75c090b21575b96879861a69bf00daa1a84515112e8571fc0c",
                  "ResolutionTime": "2020-01-21 15:29:33.654000-08:00"
                }
              ],
              "CurrentWeight": 1.0,
              "DesiredWeight": 1.0,
              "CurrentInstanceCount": 1,
              "DesiredInstanceCount": 1
            }
          ],
          "EndpointStatus": "InService",
          "CreationTime": "2020-01-21 15:29:31.760000-08:00",
          "LastModifiedTime": "2020-01-21 15:38:44.080000-08:00",
          "ResponseMetadata": {
            "RequestId": "6e946239-1aa3-4a8c-9803-226f6d19b0c7",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
              "x-amzn-requestid": "6e946239-1aa3-4a8c-9803-226f6d19b0c7",
              "content-type": "application/x-amz-json-1.1",
              "content-length": "835",
              "date": "Tue, 21 Jan 2020 23:38:44 GMT"
            },
            "RetryAttempts": 0
          }
        },
        "timestamp": "2020-01-21T23:38:44.586400Z"
      },
      "createdAt": "2020-01-21T23:26:41.575952Z",
      "lastUpdatedAt": "2020-01-21T23:26:41.576004Z"
    }


After deploy to Sagemaker, use `bentoml sagemaker get` to return and display the latest status
and other information about the deployment

.. code-block:: bash

    > bentoml sagemaker get my-first-sagemaker-deployment

    {
      "namespace": "dev",
      "name": "my-first-sagemaker-deployment",
      "spec": {
        "bentoName": "IrisClassifier",
        "bentoVersion": "20200121141808_FE78B5",
        "operator": "AWS_SAGEMAKER",
        "sagemakerOperatorConfig": {
          "region": "us-west-2",
          "instanceType": "ml.m4.xlarge",
          "instanceCount": 1,
          "apiName": "predict"
        }
      },
      "state": {
        "state": "RUNNING",
        "infoJson": {
          "EndpointName": "dev-my-first-sagemaker-deployment",
          "EndpointArn": "arn:aws:sagemaker:us-west-2:192023623294:endpoint/dev-my-first-sagemaker-deployment",
          "EndpointConfigName": "dev-my-first-sag-IrisClassifier-20200121141808-FE78B5",
          "ProductionVariants": [
            {
              "VariantName": "dev-my-first-sag-IrisClassifier-20200121141808-FE78B5",
              "DeployedImages": [
                {
                  "SpecifiedImage": "192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-sagemaker:20200121141808_FE78B5",
                  "ResolvedImage": "192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-sagemaker@sha256:cd723a363bcbad75c090b21575b96879861a69bf00daa1a84515112e8571fc0c",
                  "ResolutionTime": "2020-01-21 15:29:33.654000-08:00"
                }
              ],
              "CurrentWeight": 1.0,
              "DesiredWeight": 1.0,
              "CurrentInstanceCount": 1,
              "DesiredInstanceCount": 1
            }
          ],
          "EndpointStatus": "InService",
          "CreationTime": "2020-01-21 15:29:31.760000-08:00",
          "LastModifiedTime": "2020-01-21 15:38:44.080000-08:00",
          "ResponseMetadata": {
            "RequestId": "2a2ac5bc-8381-4d34-b283-a48b401f0955",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
              "x-amzn-requestid": "2a2ac5bc-8381-4d34-b283-a48b401f0955",
              "content-type": "application/x-amz-json-1.1",
              "content-length": "835",
              "date": "Tue, 21 Jan 2020 23:40:54 GMT"
            },
            "RetryAttempts": 0
          }
        },
        "timestamp": "2020-01-21T23:40:55.332500Z"
      },
      "createdAt": "2020-01-21T23:26:41.575952Z",
      "lastUpdatedAt": "2020-01-21T23:26:41.576004Z"
    }


Use AWS CLI to verify that the BentoService is properly deployed to Sagemaker and is ready to inference

.. code-block:: bash

    > aws sagemaker describe-endpoint --endpoint-name dev-my-first-sagemaker-deployment

    {
        "EndpointName": "dev-my-first-sagemaker-deployment",
        "EndpointArn": "arn:aws:sagemaker:us-west-2:192023623294:endpoint/dev-my-first-sagemaker-deployment",
        "EndpointConfigName": "dev-my-first-sag-IrisClassifier-20200121141808-FE78B5",
        "ProductionVariants": [
            {
                "VariantName": "dev-my-first-sag-IrisClassifier-20200121141808-FE78B5",
                "DeployedImages": [
                    {
                        "SpecifiedImage": "192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-sagemaker:20200121141808_FE78B5",
                        "ResolvedImage": "192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-sagemaker@sha256:cd723a363bcbad75c090b21575b96879861a69bf00daa1a84515112e8571fc0c",
                        "ResolutionTime": 1579649373.654
                    }
                ],
                "CurrentWeight": 1.0,
                "DesiredWeight": 1.0,
                "CurrentInstanceCount": 1,
                "DesiredInstanceCount": 1
            }
        ],
        "EndpointStatus": "InService",
        "CreationTime": 1579649371.76,
        "LastModifiedTime": 1579649924.08
    }


Use the sample data to verify the predict result from the Sagemaker deployment

.. code-block:: bash

    > aws sagemaker-runtime invoke-endpoint \
      --endpoint-name dev-my-first-sagemaker-deployment \
      --body '[[5.1, 3.5, 1.4, 0.2]]' \
      --content-type "application/json" \
      >(cat) 1>/dev/null | jq .

    [0]{
      "ContentType": "application/json",
      "InvokedProductionVariant": "dev-my-first-sag-IrisClassifier-20200121141808-FE78B5"
    }


Use `bentoml sagemaker list` to display all sagemaker deployments managed by BentoML

.. code-block:: bash

    > bentoml sagemaker list

    NAME                           NAMESPACE    LABELS    PLATFORM       STATUS    AGE
    my-first-sagemaker-deployment  dev                    aws-sagemaker  running   15 minutes and 21.14 seconds


Removing Sagemaker deployment is as easy as deploying one.  BentoML will remove any related resources from AWS as well.

.. code-block:: bash

    > bentoml sagemaker delete my-first-sagemaker-deployment

    Successfully deleted AWS Sagemaker deployment "my-first-sagemaker-deployment"


====================================================================
Deploy and manage AWS Sagemaker deployments with remote YataiService
====================================================================

BentoML recommends to use remote YataiService for managing and deploying BentoService
when you are working in a team. To deploy AWS Sagemaker deployments with remote
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




===========================================================
Deploy and manage AWS Sagemaker deployments with Kubernetes
===========================================================

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



