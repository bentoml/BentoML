Deploying to AWS SageMaker
==========================

AWS Sagemaker is a fully managed services for quickly building ML models. BentoML provides great support
for deploying BentoService to AWS Sagemaker without additional process and work from user. With BentoML,
users can enjoy the great system performance from Sagemaker with any popular ML frameworks.


Prerequisites
-------------

* An active AWS account configured on the machine with AWS CLI installed and configured

  * Install instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html
  * Configure AWS account instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

* Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install



Deploy BentoService to AWS Sagemaker
------------------------------------

This guide uses the IrisClassifier BentoService from the :doc:`Quick start guide <../quickstart>`:

.. code-block:: bash

    > git clone git@github.com:bentoml/BentoML.git
    > pip install -r ./bentoml/guides/quick-start/requirements.txt
    > python ./bentoml/guides/quick-start/main.py


Download and Install BentoML Sagemaker deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/aws-sagemaker-deploy
    > cd aws-sagemaker-deploy
    > pip install -r requirements.txt



Edit `sagemaker_config.json`  file with options for the deployment.

**Available configuration options for AWS Sagemaker deployment**

* `region`: AWS region where Sagemaker endpoint is deploying to
* `api_name`: User-defined API function name.
* `timeout`: Timeout for API request in seconds. Default is 60
* `workers`: Number of workers for the deployment
* `instance_type`: The ML computing instance type for the deployed Sagemaker endpoint. See the link for more information, https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-endpoint-config.html
* `initial_instance_count`: Number of the instances to launch initially
* `enable_data_capture`: Boolean toggle for enable Sagemaker to captures data from requests/responses and store the captured data to S3 bucket
* `data_capture_s3_prefix`: S3 bucket path for store captured data
* `data_capture_sample_percent`: The percentage of the data will be captured to the S3 bucket.

.. code-block:: bash

    > BENTO_BUNDLE=$(bentoml get Bento_Name:Bento_version --print-location -q)
    > python deploy.py $BENTO_BUNDLE my-first-sagemaker-deployment sagemaker_config.json


Get the deployment information and status

.. code-block:: bash

    > python describe.py my-first-sagemaker-depoyment


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


Delete Sagemaker deployment

.. code-block:: bash

    > python delete.py my-first-sagemaker-deployment


Migrating to BentoML Sagemaker deployment tool
----------------------------------------------

1. Delete the previous deployment use BentoML CLI tool

.. code-block:: bash

    > bentoml sagemaker delete DEPLOYMENT_NAME


2. Download and Install BentoML Sagemaker deployment tool

.. code-block:: bash

    > git clone https://github.com/bentoml/aws-sagemaker-deploy
    > cd aws-sagemaker-deploy
    > pip install -r requirements.txt

3. Deploy to Sagemaker with deployment tool

.. code-block:: bash

    > BENTO_BUNDLE=$(bentoml get Bento_Name:Bento_version --print-location -q)
    > python deploy.py $BENTO_BUNDLE my_deployment sagemaker_config.json
