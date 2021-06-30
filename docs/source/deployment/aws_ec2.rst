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
    > python deploy.py $BENTO_BUNDLE_PATH my-first-ec2-deployment ec2_config.json

    # Sample output
    Creating S3 bucket for cloudformation
    Build and push image to ECR
    Generate CF template
    Build CF template
    Deploy EC2


Get EC2 deployment information and status:

.. code-block:: bash

    > python describe.py my-first-ec2-deployment ec2_config.json

    # Sample output
    {
      "InstanceDetails": [
        {
          "instance_id": "i-03ff2d1b9b717a109",
          "endpoint": "3.101.38.18",
          "state": "InService",
          "health_status": "Healthy"
        }
      ],
      "Endpoints": [
        "3.101.38.18:5000/"
      ],
      "S3Bucket": "my-ec2-deployment-storage",
      "TargetGroup": "arn:aws:elasticloadbalancing:us-west-1:192023623294:targetgroup/my-ec-Targe-3G36XKKIJZV9/d773b029690c84d3",
      "Url": "http://my-ec2-deployment-elb-2078733703.us-west-1.elb.amazonaws.com"
    }


Tests the deployed service with sample dataset:

.. code-block:: bash

    > curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      https://ps6f0sizt8.execute-api.us-west-2.amazonaws.com/predict

    # Sample output
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


Delete EC2 deployment

.. code-block:: bash

    > python delete.py my-first-ec2-deployment

    # Sample output
    Delete CloudFormation Stack my-ec2-deployment-stack
    Delete ECR repo my-ec2-deployment-repo
    Delete S3 bucket my-ec2-deployment-storage


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