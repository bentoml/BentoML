Deploying to AWS EC2
=======================


AWS EC2 is a great choice for deploying containerized and load balanced services in the cloud.
It's ability to autoscale and automated health checking features make it attractive to
users who want to reduce cost and want to horizontally scale base on traffic.


Prerequisites
-------------

* An active AWS account configured on the machine with AWS CLI installed and configured

  * Install instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html
  * Configure AWS account instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

* AWS SAM CLI tool

  * Install instruction: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html

* Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install


AWS EC2 deployment and management with BentoML
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


BentoML has great support for AWS EC2. You can deploy, update and delete
deployment with single command, and customize deployment to fit your needs with parameters
such as `instance type`,`scaling capacities`

To deploy BentoService to AWS EC2, use `bentoml lambda deploy` command.
Deployment name and bento service name:version tag is required.

.. code-block:: bash

    > bentoml ec2 deploy my-first-ec2-deployment -b IrisClassifier:20200121141808_FE78B5

    Deploying EC2 deployment -[2020-01-21 14:37:16,838] INFO - Building project
    [2020-01-21 14:38:52,826] INFO - Containerzing service
    [2020-01-21 14:39:18,834] INFO - Deploying
    [2020-01-21 14:40:09,265] INFO - ApplyDeployment (my-first-ec2-deployment, namespace dev) succeeded
    Successfully created AWS EC2 deployment my-first-ec2-deployment
    {
    "namespace": "dev",
    "name": "my-first-ec2-deployment",
    "spec": {
        "bentoName": "IrisClassifier",
        "bentoVersion": "20200121141808_FE78B5",
        "operator": "AWS_EC2",
        "awsEc2OperatorConfig": {
        "region": "ap-south-1",
        "instanceType": "t2.micro",
        "amiId": "/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2",
        "autoscaleMinCapacity": 1,
        "autoscaleDesiredCapacity": 1,
        "autoscaleMaxCapacity": 1
        }
    },
    "state": {},
    "createdAt": "2020-10-24T06:52:18.003580Z",
    "lastUpdatedAt": "2020-10-24T06:52:18.003626Z"
    }



BentoML helps you manage the entire process of deploying your BentoService bundle to EC2.
Verify the deployed resources with AWS CLI tool:

.. code-block:: bash

    > aws cloudformation describe-stacks

    {
        "Stacks": [
            {
              "StackId": "arn:aws:cloudformation:ap-south-1:752014255238:stack/btml-stack-dev-my-first-ec2-deployment/a9d08770-1d10-11eb-bc31-028b9ab9a492",
              "StackName": "btml-stack-dev-my-first-ec2-deployment",
              "ChangeSetId": "arn:aws:cloudformation:ap-south-1:752014255238:changeSet/samcli-deploy1604324294/ac735ad1-6080-43d2-9e9f-2484563d31c8",
              "Description": "BentoML load balanced template",
              "Parameters": [
                  {
                      "ParameterKey": "AmazonLinux2LatestAmiId",
                      "ParameterValue": "/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2",
                      "ResolvedValue": "ami-0e306788ff2473ccb"
                  }
              ],
              "CreationTime": "2020-11-02T13:38:17.257000+00:00",
              "LastUpdatedTime": "2020-11-02T13:38:22.926000+00:00",
              "RollbackConfiguration": {},
              "StackStatus": "CREATE_COMPLETE",
              "DisableRollback": false,
              "NotificationARNs": [],
              "Capabilities": [
                  "CAPABILITY_IAM"
              ],
              "Outputs": [
                  {
                      "OutputKey": "AutoScalingGroup",
                      "OutputValue": "btml-stack-dev-my-first-ec2-deployment-AutoScalingGroup-GTO3DXSAZSWK",
                      "Description": "Autoscaling group name"
                  },
                  {
                      "OutputKey": "S3Bucket",
                      "OutputValue": "btml-752014255238-dev",
                      "Description": "Bucket to store sam artifacts"
                  },
                  {
                      "OutputKey": "TargetGroup",
                      "OutputValue": "arn:aws:elasticloadbalancing:ap-south-1:752014255238:targetgroup/btml-Targe-1PBR6D87075CO/b3f6c6296ee51758",
                      "Description": "Target group for load balancer"
                  },
                  {
                      "OutputKey": "Url",
                      "OutputValue": "http://btml-LoadB-1QA80SD51INOM-516888199.ap-south-1.elb.amazonaws.com",
                      "Description": "URL of the bento service"
                  }
              ],
              "Tags": [],
              "DriftInformation": {
                  "StackDriftStatus": "NOT_CHECKED"
              }
          },

        ]
    }

Tests the deployed service with sample dataset:

.. code-block:: bash

    > curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      https://ps6f0sizt8.execute-api.us-west-2.amazonaws.com/predict

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

    > bentoml ec2 get my-first-ec2-deployment

    {
        "namespace": "dev",
        "name": "deploy-103",
        "spec": {
            "bentoName": "IrisClassifier",
            "bentoVersion": "20201015064204_282D00",
            "operator": "AWS_EC2",
            "awsEc2OperatorConfig": {
            "region": "ap-south-1",
            "instanceType": "t2.micro",
            "amiId": "/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2",
            "autoscaleMinCapacity": 1,
            "autoscaleDesiredCapacity": 1,
            "autoscaleMaxCapacity": 1
            }
        },
        "state": {
            "state": "RUNNING",
            "infoJson": {
            "InstanceDetails": [
                {
                "instance_id": "i-0a8ebeb105e941257",
                "endpoint": "65.0.11.248",
                "state": "InService",
                "health_status": "Healthy"
                }
            ],
            "Endpoints": [
                "65.0.11.248:5000/predict"
            ],
            "S3Bucket": "btml-752014255238-dev",
            "TargetGroup": "arn:aws:elasticloadbalancing:ap-south-1:752014255238:targetgroup/btml-Targe-II1UG5WJJVPV/b2d6137a7485a45e",
            "Url": "http://btml-LoadB-9K2SGQEFUKFK-432766095.ap-south-1.elb.amazonaws.com"
            }
        },
        "createdAt": "2020-10-24T06:56:08.974179Z",
        "lastUpdatedAt": "2020-10-24T06:56:08.974212Z"
        }


Use `bentoml ec2 list` to have a quick glance of all of the AWS EC2 deployments

.. code-block:: bash

    > bentoml ec2 list

    NAME                        NAMESPACE    LABELS    PLATFORM                               STATUS    AGE
    my-first-ec2-deployment     dev          aws-ec2   IrisClassifier:20201015064204_282D00   running   10 minutes and 3.72 seconds


Removing a EC2 deployment is also very easy.  Calling `bentoml ec2 delete` command will delete the all resources from aws.

.. code-block:: bash

    > bentoml ec2 delete my-first-ec2-deployment

    Successfully deleted AWS EC2 deployment "my-first-ec2-deployment"


=================================================================
Deploy and manage AWS EC2 deployments with remote YataiService
=================================================================

BentoML recommends to use remote YataiService for managing and deploying BentoService
when you are working in a team. To deploy AWS EC2 deployments with remote
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
Deploy and manage AWS EC2 deployments with Kubernetes
========================================================

Create a Kubernetes secret with the the AWS credentials.

Generate base64 strings from the AWS credentials from your AWS config file.

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


=================================================================
Permissions required on AWS for deployment
=================================================================

    * SSM:GetParameters
    * AmazonEC2FullAccess
    * AmazonEC2ContainerRegistryFullAccess 
    * AmazonS3FullAccess
    * IAMFullAccess
    * AmazonVPCFullAccess
    * AWSCloudFormationFullAccess 
    * CloudWatchFullAccess
    * ElasticLoadBalancingFullAccess 
    * AutoScalingFullAccess  


.. spelling::

    analytics
    SSM
    GetParameters