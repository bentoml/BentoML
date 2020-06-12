Deploying to AWS ECS(Elastic Container Service)
===============================================


AWS ECS (elastic container service) is a fully managed container orchestration service.
With AWS Fargate, a serverless compute engine for containers, ECS provides the benefit
of AWS Lambda without sacrificing computing performance. It is great for running more
advanced ML prediction service that require more computing power compare to AWS Lambda,
while still want to take advantage of the benefits that AWS Lambda brings.

This guide demonstrates how to serve a scikit-learn based iris classifier model with
BentoML on AWS ECS. The same deployment steps are also applicable for models
trained with other machine learning frameworks, see more BentoML examples :doc:`here <../examples>`.

Prerequisites
-------------

* An active AWS account configured on the machine with AWS CLI installed and configured

  * Install instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html
  * Configure AWS account instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

* Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install

* AWS ECS CLI tool

  * Install instruction: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_CLI_installation.html

* Python 3.6 or above with `scikit-learn` and `bentoml` installed

  *  .. code-block:: bash

        pip install bentoml scikit-learn




AWS ECS deployment with BentoML
-------------------------------------------------

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
            "inputType": "DataframeInput",
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

=============================================
Dockerize BentoML model server for deployment
=============================================

In order to create ECS deployment, the model server need to be containerized and push to
a container registry. Amazon Elastic Container Registry (ECR) is a fully-managed Docker
container registry that makes it easy for developers to store, manage, and deploy Docker
container images.

Docker login with AWS ECR

.. code-block:: bash

    $ aws ecr get-login --region us-west-2 --no-include-email

    # Sample output

    docker login -u AWS -p eyJ.................OOH https://account_id.dkr.ecr.us-west-2.amazonaws.com

Copy the output from previous step and run it in the terminal

.. code-block:: bash

    $ docker login -u AWS -p eyJ.................OOH https://account_id.dkr.ecr.us-west-2.amazonaws.com

    # Sample output

    Login Succeeded

Create AWS ECR repository

.. code-block:: bash

    $ aws ecr create-repository --repository-name irisclassifier-ecs

    # Sample output

    {
        "repository": {
            "repositoryArn": "arn:aws:ecr:us-west-2:192023623294:repository/irisclassifier-ecs",
            "registryId": "192023623294",
            "repositoryName": "irisclassifier-ecs",
            "repositoryUri": "192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-ecs",
            "createdAt": 1576542447.0,
            "imageTagMutability": "MUTABLE",
            "imageScanningConfiguration": {
                "scanOnPush": false
            }
        }
    }


.. code-block:: bash

    # Install jq, the command-line JSON processor: https://stedolan.github.io/jq/download/
    $ saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")
    $ docker build --tag=192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-ecs $saved_path

    # Sample output

    Step 1/12 : FROM continuumio/miniconda3:4.7.12
    ...
    ...
    ...
    Successfully built 19d21c608b08
    Successfully tagged 192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-ecs:latest

Push the built docker image to AWS ECR

.. code-block:: bash

    $ docker push 192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-ecs

    # Sample output

    The push refers to repository [192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-ecs]
    ...
    ...
    785a656a85507b3717c83e8a1d4c901605c4fa301364c7c18fc30346 size: 2225


==============================
Prepare AWS for ECR deployment
==============================

--------------
Setup IAM role
--------------

Create `task-execution-assume-role.json`

.. code-block::

    $ cat task-execution-assume-role.json

    # Sample output

    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "",
          "Effect": "Allow",
          "Principal": {
            "Service": "ecs-tasks.amazonaws.com"
          },
          "Action": "sts:AssumeRole"
        }
      ]
    }


Create IAM role

.. code-block::

    $ aws iam --region us-west-2 create-role --role-name ecsTaskExecutionRole \
      --assume-role-policy-document file://task-execution-assume-role.json

    # Sample output

    {
        "Role": {
            "Path": "/",
            "RoleName": "ecsTaskExecutionRole",
            "RoleId": "AROASZNL76Z7C7Q7SZJ4D",
            "Arn": "arn:aws:iam::192023623294:role/ecsTaskExecutionRole",
            "CreateDate": "2019-12-17T01:04:08Z",
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "",
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "ecs-tasks.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
        }
    }


.. code-block:: bash

    aws iam --region us-west-2 attach-role-policy --role-name ecsTaskExecutionRole \
      --policy-arn arn:aws:iam:aws:policy/service-role/AmazonECSTaskExecutionRolePolicy


=================
Configure ECR CLI
=================

Create ECR CLI profile

.. code-block:: bash

    ecs-cli configure profile --access-key AWS_ACCESS_KEY_ID --secret-key AWS_SECRET_ACCESS_KEY --profile-name tutorial-profile


Create ECR cluster profile configuration

.. code-block:: bash

    ecs-cli configure --cluster tutorial --default-launch-type FARGATE --config-name tutorial --region us-west-2


==================================
Prepare ECR cluster for deployment
==================================

Start ECR cluster with the ecr profile we created in the earlier step

.. code-block:: bash

    $ ecs-cli up --cluster-config tutorial --ecs-profile tutorial-profile

    # Sample output

    INFO[0001] Created cluster                               cluster=tutorial region=us-west-2
    INFO[0002] Waiting for your cluster resources to be created...
    INFO[0002] Cloudformation stack status                   stackStatus=CREATE_IN_PROGRESS
    INFO[0063] Cloudformation stack status                   stackStatus=CREATE_IN_PROGRESS
    VPC created: vpc-0465d14ba04402f80
    Subnet created: subnet-0d23851806f3db403
    Subnet created: subnet-0dece5451f1a3b8b2
    Cluster creation succeeded.

Use the VPC id from previous command to get security group ID

.. code-block:: bash

    $ aws ec2 describe-security-groups --filters Name=vpc-id,Values=vpc-0465d14ba04402f80 \
      --region us-west-2

    # Sample output

    {
        "SecurityGroups": [
            {
                "Description": "default VPC security group",
                "GroupName": "default",
                "IpPermissions": [
                    {
                        "IpProtocol": "-1",
                        "IpRanges": [],
                        "Ipv6Ranges": [],
                        "PrefixListIds": [],
                        "UserIdGroupPairs": [
                            {
                                "GroupId": "sg-0258b891f053e077b",
                                "UserId": "192023623294"
                            }
                        ]
                    }
                ],
                "OwnerId": "192023623294",
                "GroupId": "sg-0258b891f053e077b",
                "IpPermissionsEgress": [
                    {
                        "IpProtocol": "-1",
                        "IpRanges": [
                            {
                                "CidrIp": "0.0.0.0/0"
                            }
                        ],
                        "Ipv6Ranges": [],
                        "PrefixListIds": [],
                        "UserIdGroupPairs": []
                    }
                ],
                "VpcId": "vpc-0465d14ba04402f80"
            }
        ]
    }

Use security group ID from previous command

.. code-block:: bash

    aws ec2 authorize-security-group-ingress --group-id sg-0258b891f053e077b --protocol tcp \
    --port 5000 --cidr 0.0.0.0/0 --region us-west-2


=====================================
Deploying BentoService to ECR cluster
=====================================

Create `docker-compose.yaml` file, use the image tag from previous steps

.. code-block:: yaml

    version: '3'
    services:
      web:
        image: 192023623294.dkr.ecr.us-west-2.amazonaws.com/irisclassifier-ecs
        ports:
          - "5000:5000"
        logging:
          driver: awslogs
          options:
            awslogs-group: irisclassifier-aws-ecs
            awslogs-region: us-west-2
            awslogs-stream-prefix: web


Compose `ecs-params.yaml` with subnets information from starting up ECS cluster, and security group id from describe security group

.. code-block:: yaml

    version: 1
    task_definition:
      task_execution_role: ecsTaskExecutionRole
      ecs_network_mode: awsvpc
      task_size:
        mem_limit: 0.5GB
        cpu_limit: 256
    run_params:
      network_configuration:
        awsvpc_configuration:
          subnets:
            - subnet-0d23851806f3db403
            - subnet-0dece5451f1a3b8b2
          security_groups:
            - sg-0258b891f053e077b
          assign_public_ip: ENABLED


After create `ecs-params.yaml`, we can deploy our BentoService to the ECS cluster

.. code-block:: bash

    $ ecs-cli compose --project-name tutorial-bentoml-ecs service up --create-log-groups \
      --cluster-config tutorial --ecs-profile tutorial-profile

    # Sample output

    INFO[0000] Using ECS task definition                     TaskDefinition="tutorial-bentoml-ecs:1"
    WARN[0001] Failed to create log group sentiment-aws-ecs in us-west-2: The specified log group already exists
    INFO[0001] Updated ECS service successfully              desiredCount=1 force-deployment=false service=tutorial-bentoml-ecs
    INFO[0017] (service tutorial-bentoml-ecs) has started 1 tasks: (task ecd119f0-b159-42e6-b86c-e6a62242ce7a).  timestamp="2019-12-17 01:05:23 +0000 UTC"
    INFO[0094] Service status                                desiredCount=1 runningCount=1 serviceName=tutorial-bentoml-ecs
    INFO[0094] (service tutorial-bentoml-ecs) has reached a steady state.  timestamp="2019-12-17 01:06:40 +0000 UTC"
    INFO[0094] ECS Service has reached a stable state        desiredCount=1 runningCount=1 serviceName=tutorial-bentoml-ecs


Now, after creating the service, we can use `ecs-cli service ps` command to check the service's status

.. code-block:: bash

    $ ecs-cli compose --project-name tutorial-bentoml-ecs service ps \
      --cluster-config tutorial --ecs-profile tutorial-profile

    # Sample output

    Name                                      State    Ports                        TaskDefinition          Health
    ecd119f0-b159-42e6-b86c-e6a62242ce7a/web  RUNNING  34.212.49.46:5000->5000/tcp  tutorial-bentoml-ecs:1  UNKNOWN


====================================
Testing ECS service with sample data
====================================

.. code-block:: bash

    $ curl -i \
      --request POST \
      --header "Content-Type: application/json" \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      http://34.212.49.46:5000/predict

    [0]


===========================
Clean up AWS ECS Deployment
===========================

Delete the service on AWS ECS

.. code-block:: bash

    $ ecs-cli compose --project-name tutorial-bentoml-ecs service down --cluster-config tutorial \
      --ecs-profile tutorial-profile

    # Sample output

    INFO[0000] Updated ECS service successfully              desiredCount=0 force-deployment=false service=tutorial-bentoml-ecs
    INFO[0000] Service status                                desiredCount=0 runningCount=1 serviceName=tutorial-bentoml-ecs
    INFO[0016] Service status                                desiredCount=0 runningCount=0 serviceName=tutorial-bentoml-ecs
    INFO[0016] (service tutorial-bentoml-ecs) has stopped 1 running tasks: (task ecd119f0-b159-42e6-b86c-e6a62242ce7a).  timestamp="2019-12-17 01:15:37 +0000 UTC"
    INFO[0016] ECS Service has reached a stable state        desiredCount=0 runningCount=0 serviceName=tutorial-bentoml-ecs
    INFO[0016] Deleted ECS service                           service=tutorial-bentoml-ecs
    INFO[0016] ECS Service has reached a stable state        desiredCount=0 runningCount=0 serviceName=tutorial-bentoml-ecs


Shutting down the AWS ECS cluster

.. code-block:: bash

    $ ecs-cli down --force --cluster-config tutorial --ecs-profile tutorial-profile

    # Sample output

    INFO[0001] Waiting for your cluster resources to be deleted...
    INFO[0001] Cloudformation stack status                   stackStatus=DELETE_IN_PROGRESS
    INFO[0062] Deleted cluster                               cluster=tutorial
