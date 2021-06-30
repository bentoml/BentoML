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


Deploy BentoService to AWS EC2
------------------------------

This guide uses the IrisClassifier BentoService from the :doc:`Quick start guide <../quickstart>`:

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    pip install -r ./bentoml/guides/quick-start/requirements.txt
    python ./bentoml/guides/quick-start/main.py


Download and Install BentoML EC2 deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/aws-ec2-deploy
    > cd aws-ec2-deploy
    > pip install -r requirements.txt


Edit the deployment options `ec2_config.json` file

**Available configuration options for AWS EC2 deployment**

* `region`: AWS region for EC2 deployment
* `ec2_auto_scale`:
    * `min_size`:  The minimum number of instances for the auto scale group. Default is 1
    * `desired_capacity`: The desired capacity for the auto scale group. Auto Scaling group will start by launching as many instances as are specified for desired capacity. Default is 1
    * `max_size`: The maximum number of instances for the auto scale group. Default is 1
* `instance_type`: Instance type for the EC2 deployment. See https://aws.amazon.com/ec2/instance-types/ for more info.
* `ami_id`: The Amazon machine image (AMI) used for launching EC2 instance. Default is `/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2`. See https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html for more information.
* `elb`:
    * `health_check_interval_seconds`: The approximate interval, in seconds, between health checks of an individual instance. Valid Range: Minimum value of 5. Maximum value of 300.
    * `health_check_path.`: The URL path for health check. Default is `/healthz`
    * `health_check_port`: Health check port. Default is `5000`
    * `health_check_timeout_seconds`: The amount of time, in seconds, during which no response means a failed health check.
    * `healthy_threshold_count`: The number of consecutive health checks successes required before moving the instance to the Healthy state. Valid Range: Minimum value of 2. Maximum value of 10.


.. code-block:: bash

    > BENTO_BUNDLE_PATH=$(bentoml get IrisClassifier:latest --print-location -q)
    > python deploy.py my-first-ec2-deployment $BENTO_BUNDLE_PATH ec2_config.json


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


Get the deployment information and status

.. code-block:: bash

    > python describe.py my-first-ec2-deployment

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



Delete EC2 deployment

.. code-block:: bash

    > python delete.py my-first-ec2-deployment


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


Migrating to BentoML EC2 deployment tool
----------------------------------------

1. Delete the previous deployment use BentoML CLI tool

.. code-block:: bash

    > bentoml ec2 delete DEPLOYMENT_NAME


2. Download and Install BentoML Lambda deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/aws-ec2-deploy
    > cd aws-ec2-deploy
    > pip install -r requirements.txt

3. Deploy to EC2 with deployment tool

.. code-block:: bash

    > BENTO_BUNDLE=$(bentoml get Bento_Name:Bento_version --print-location -q)
    > python deploy.py $BENTO_BUNDLE my_deployment ec2_config.json


.. spelling::

    analytics
    SSM
    GetParameters