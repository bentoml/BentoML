Building an AWS SageMaker End-to-end Workflow with BentoML
=====================================================================

This tutorial provides an end-to-end guide to using BentoML with AWS SageMaker -- a machine learning model training platform. It demonstrates the workflow of integrating BentoML with SageMaker, including: setting up a SageMaker notebook instance, model training, creating an S3 bucket, uploading the BentoService bundle into S3, and deploying the BentoML packaged model to SageMaker as an API endpoint using the BentoML CLI tool.

For demonstration, this tutorial uses the IMDB movie review sentiment dataset with BERT and Tensorflow 2.0. (please note: the following model is a modification of the `original version <https://github.com/kpe/bert-for-tf2/blob/master/examples/gpu_movie_reviews.ipynb>`_)

Prerequisites
-------------
* An active AWS account configured on the machine with AWS CLI installed and configurated

    * Install instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html

    * Configure AWS account instruction: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html

* Docker 

  * Install instruction: https://docs.docker.com/install

* Python 3.6 or above and required packages `bentoml` and `bert-for-tf2`:

  * .. code-block:: bash

        pip install bentoml bert-for-tf2


1 Create a SageMaker notebook instance
---------------------------------------

For model training in SageMaker, log in to the AWS management console and navigate to SageMaker. From the SageMaker dashboard, select Notebook instances. Go ahead enter a notebook name and select the instance type

.. image:: _static/img/create-notebook-instance.png

.. image:: _static/img/gcloud-setting.png

Next,under **Permissions and encryption** , select **Create a new role** or **choosing an existing role** . This allows both the notebook instance and user to access and upload data to Amazon S3. Then, select Any S3 bucket, which allows your SageMaker to access all S3 buckets in your account.

.. image:: _static//img/create-IAM-role.png

After the notebook instance is created, the status will change from pending to **InService** . Select Open Jupyter under Actions, and choose **Conda_python 3** under New tab to launch the Jupyter notebook within SageMaker.

.. note::

   SageMaker also provides a local model through pip install SageMaker.

Finally to prepare for the model training, let's import some libraries -- Boto3 and SageMaker and set up the IAM role. Boto3 is the AWS SDK for Python, which makes it easier to integrate our model with AWS services such as Amazon S3

.. code-block:: python

    import boto3, sagemaker
    from sagemaker import get_execution_role

    # Define IAM role
    role = get_execution_role()
    prefix = 'sagemaker/bert-moviereview-bento'
    my_region = boto3.session.Session().region_name # set the region of the instance    

In this step, we will create an S3 bucket named movie-review-dataset to store the dataset. Users could click on the bucket name and upload the dataset directly into S3. Alternatively, for cost-efficiency, users could train the model locally using the SageMaker local mode

.. code-block:: python

    bucket_name = 'movie-review-dataset'
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=bucket_name)

    # sample output

    s3.Bucket(name='movie-review-dataset')


.. image:: _static/img/create-s3-bucket.png


2 Model Training -- Movie review sentiment with BERT and TensorFlow 2
---------------------------------------------------------------------

The second step of this tutorial is model training. We will be using the IMDB movie review dataset to create a sentiment analysis model which contains 25K positive and negative movie reviews each.

Below is the model summary. Please checkout :code:`bentoml/gallery/end-to-end-sagemaker-depoyment` for more details on model training.

.. code-block:: python

    model = tf.keras.models.load_model('saved_model/my_model')

    model.summary()

    # sample output

        Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_ids (InputLayer)       [(None, 128)]             0         
    _________________________________________________________________
    bert (BertModelLayer)        (None, 128, 768)          108890112 
    _________________________________________________________________
    lambda (Lambda)              (None, 768)               0         
    _________________________________________________________________
    dropout (Dropout)            (None, 768)               0         
    _________________________________________________________________
    dense (Dense)                (None, 768)               590592    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 768)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 1538      
    =================================================================
    Total params: 109,482,242
    Trainable params: 109,482,242
    Non-trainable params: 0


3 BentoML SageMaker API Endpoints Deployment
---------------------------------------------

In this section, we will demonstrate on using BentoML to build production-ready API endpoints and deploy it to AWS SageMaker. The core steps are as follows:

1. Create a BentoML service file for model prediction 
2. Create and save a BentoMl packaged model called BentoService bundle for model deployment
3. Upload the BentoService bundle to cloud storage like S3 (optional)
4. Use Bento CLI and its web UI for local testing
5. Deploy AWS SageMaker API endpoints through Bento CLI
6. Use AWS boto3 SDK or AWS CLI for endpoints testing

.. note::

    for AWS SageMaker deployment, you will need the following prerequisites as stated before: 

    * Install and configure the AWS CLI 
    * Install Docker

    for more information, please `click here <https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html>`_ 

================================================
3.1 Create a BentoML Service File for Prediction
================================================

First, let's create a prediction service file using BentoML. The three main BentoML concepts are:

1. Define the bentoml service environment
2. Define the model artifacts based on the ML frameworks used for the trained model
3. Choose the relevant input adapters (formerly handlers) for the API

Note: BentoML supports a variety of major ML frameworks and input data format. For more details, please check available model artifacts `here <https://docs.bentoml.org/en/latest/api/artifacts.html>`_ and adapters `here <https://docs.bentoml.org/en/latest/api/adapters.html>`_ 

For defining the BentoML service environment and trouble-shooting, you would also use  :code:`auto_pip_dependencies= True` or pass the BentoML generated requirement.txt through  :code:`@bentoml.env(requirements_tex_file ='./requirements.txt')`

.. code-block:: python

    %%writefile bentoml_service.py

    import tensorflow as tf
    import numpy as np
    import pandas as pd

    import bentoml
    from bentoml.artifact import (TensorflowSavedModelArtifact, PickleArtifact)
    from bentoml.adapters import DataframeInput

    CLASSES  = ['negative','positive']
    max_seq_len = 128

    try:
        tf.config.set_visible_devices([],'GPU') 
    except:
        pass

    #define bentoml service environment
    @bentoml.env(pip_dependencies=['tensorflow','bert','bert-for-tf2','numpy==1.18.1','pandas==1.0.1'])
    #define model artifacts
    @bentoml.artifacts([TensorflowSavedModelArtifact('model'), PickleArtifact('tokenizer')])

    class Service(bentoml.BentoService):

        def tokenize(self, inputs: pd.DataFrame):
            tokenizer = self.artifacts.tokenizer
            if isinstance(inputs, pd.DataFrame):
                inputs = inputs.to_numpy()[:, 0].tolist()
            else: 
                inputs = inputs.tolist()
            pred_tokens = map(tokenizer.tokenize, inputs)
            pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
            pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
            pred_token_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), pred_token_ids)
            pred_token_ids = tf.constant(list(pred_token_ids), dtype=tf.int32)
            return pred_token_ids
        
        # choose dataframe input adapter 
        @bentoml.api(input = DataframeInput(), md_max_latency = 300, mb_max_batch_size=20)
        def predict(self, inputs):
            model = self.artifacts.model
            pred_token_ids = self.tokenize(inputs)
            res = model(pred_token_ids).numpy().argmax(axis =-1)
            return [CLASSES[i] for i in res]
    
    #Sample output
    Overwriting bentoml_service.py

========================================
3.2 Create and Save BentoService Bundle
========================================

The following few lines of codes demonstrate the simplicity and time-saving benefits of using BentoML. Here, we first create a BentoService instance and then use the BentoService **pack method** to bundle our trained movie review model together. Finally, we use the BentoService **save method** to save this BentoService bundle, which is now ready for inference. This process eliminates the needs for reproducing the same prediction service for testing and production environment - making it easier for data science teams to deploy their models.

By default, the BentoService bundle is saved under  :code:`~/bentoml/repository/directory`. Users could also modify the model repository through BentoML's standalone component  :code:`YataiService`, for more information, please visit `here <https://docs.bentoml.org/en/latest/concepts.html#model-management>`_ 

.. code-block:: python

    from bentoml_service import Service

    #create a service instance for the movie review model
    bento_svc = Service()

    # pack model artifacts
    bento_svc.pack('model',model)
    bento_svc.pack('tokenizer',tokenizer)

    #save the prediction service for model serving 
    saved_path = bento_svc.save()

    # sample output

    INFO:tensorflow:Assets written to: /private/var/folders/vn/bytl5x0n3vgg1vmg7n6qkqtc0000gn/T/bentoml-temp-35n_doz7/Service/artifacts/model_saved_model/assets
    [2020-06-25 19:57:01,302] INFO - Detect BentoML installed in development model, copying local BentoML module file to target saved bundle path
    running sdist
    running egg_info
    writing BentoML.egg-info/PKG-INFO
    writing dependency_links to BentoML.egg-info/dependency_links.txt
    writing entry points to BentoML.egg-info/entry_points.txt
 
    ...
    ...
    
    UPDATING BentoML-0.8.1+0.g5b6bd29.dirty/bentoml/_version.py
    set BentoML-0.8.1+0.g5b6bd29.dirty/bentoml/_version.py to '0.8.1+0.g5b6bd29.dirty'
    Creating tar archive
    removing 'BentoML-0.8.1+0.g5b6bd29.dirty' (and everything under it)
    [2020-06-25 19:57:03,958] INFO - BentoService bundle 'Service:20200625195616_62D0DB' saved to: /Users/amy/bentoml/repository/Service/20200625195616_62D0DB


=================================
Upload BentoService Bundle to S3
=================================

As mentioned earlier, BentoML also provides ways to change the model repository - allowing data science teams to share the BentoService bundle easily for better collaborations. One way is by uploading it to the cloud services such as AWS S3. Using the same scripts as above and passing the S3 bucket URL into  :code:`.save()` , it will deploy the BentoService bundle directly into the S3 movie-review-dataset bucket we created earlier.

.. code-block:: python

    from bentoml_service import Service

    #create a service instance for the movie review model
    bento_svc = Service()
 
    # pack model artifacts
    bento_svc.pack('model',model)
    bento_svc.pack('tokenizer',tokenizer)

    #save the prediction service to aws S3
    saved_path = bento_svc.save(''s3://movie-review-dataset/'')

.. image:: _static/img/show-saved-bentoservice-in-s3.png


================================
3.3 Show Existing BentoServices
================================

Using the BentoML CLI, we can see a list of BentoService generated here

.. code-block:: bash

    > bentoml list

    #sample output

        BENTO_SERVICE                         AGE                 APIS                                   ARTIFACTS
    Service:20200625195616_62D0DB         29.09 seconds       predict<DataframeInput:DefaultOutput>  model<TensorflowSavedModelArtifact>, tokenizer<PickleArtifact>
    Service:20200622153915_614FE2         3 days and 4 hours  predict<DataframeInput:DefaultOutput>  model<TensorflowSavedModelArtifact>, tokenizer<PickleArtifact>
    Service:20200622113634_A6EFDD         3 days and 8 hours  predict<DataframeInput:DefaultOutput>  model<TensorflowSavedModelArtifact>, tokenizer<PickleArtifact>
    IrisClassifier:20200615204826_CAA9DD  1 week and 2 days   predict<DataframeInput:DefaultOutput>  model<SklearnModelArtifact>
    IrisClassifier:20200615194906_60F775  1 week and 3 days   predict<DataframeInput:DefaultOutput>  model<SklearnModelArtifact>


=================================================
3.4.1 Test REST API Locally -- Online API Serving
=================================================

Before deploying the model to AWS SageMaker, we could test it locally first using the BentoML CLI. By using  :code:`bentoml serve`, it provides a near real-time prediction via API endpoints.

.. image:: _static/img/bento-web-ui.png

.. code-block:: bash

    > bentoml serve Service:20200702134432_033DAB  

    # sample output


    2020-06-26 13:43:49.634673: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    * Serving Flask app "Service" (lazy loading)
    * Environment: production
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    127.0.0.1 - - [26/Jun/2020 13:44:08] "GET / HTTP/1.1" 200 -
    127.0.0.1 - - [26/Jun/2020 13:44:09] "GET /static/swagger-ui.css HTTP/1.1" 200 -
    127.0.0.1 - - [26/Jun/2020 13:44:09] "GET /static/swagger-ui-bundle.js HTTP/1.1" 304 -
    127.0.0.1 - - [26/Jun/2020 13:44:09] "GET /docs.json HTTP/1.1" 200 -
    127.0.0.1 - - [26/Jun/2020 13:44:39] "POST /predict HTTP/1.1" 200 -


.. image:: _static/img/bento-serve-testing.png


====================================================
3.4.2 Test REST API Locally -- Offline Batch Serving
====================================================

Alternatively, we could also use  :code:`bentoml run` for local testing. BentoML provides many other model serving methods, such as: adaptive micro-batching, edge serving,and programmatic access. Please visit `here <https://docs.bentoml.org/en/latest/concepts.html#model-serving>`_ 

.. code-block:: bash

    > bentoml run Service:20200702134432_033DAB   predict --input '["the acting was a bit lacking."]'

    # sample output

    2020-06-25 20:00:04.460780: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    ['negative']


===========================
3.5 Deploy to AWS SageMaker
===========================

Finally, we are ready to deploy our BentoML packaged model to AWS SageMaker. We need to pass the deployment name, the BentoService name and the API name. Depending on the size of the BentoService generated, the deployment for this tutorial took about 30 mins.

.. code-block:: bash

    > bentoml sagemaker deploy sagemaker-moviereview-deployment -b Service:20200702134432_033DAB  --api-name predict

    # sample output

    Deploying Sagemaker deployment /[2020-06-25 20:16:14,382] INFO - Step 1/9 : FROM bentoml/model-server:0.8.1
    [2020-06-25 20:16:14,383] INFO - 

    [2020-06-25 20:16:14,383] INFO -  ---> e326316eaf10

    [2020-06-25 20:16:14,383] INFO - Step 2/9 : ENV PORT 8080
    [2020-06-25 20:16:14,384] INFO - 

    ...
    ...

    /[2020-06-25 20:18:34,080] INFO - Successfully built 1e52bd886529

    [2020-06-25 20:18:34,085] INFO - Successfully tagged 899399195124.dkr.ecr.us-east-1.amazonaws.com/service-sagemaker:20200625195616_62D0DB

    \[2020-06-25 20:53:09,669] INFO - ApplyDeployment (bert-moviereview-sagemaker, namespace dev) succeeded
    
    Successfully created AWS Sagemaker deployment bert-moviereview-sagemaker
    {
    "namespace": "dev",
    "name": "sagemaker-moviereview-sagemaker",
    "spec": {
        "bentoName": "Service",
        "bentoVersion": "20200702134432_033DAB",
        "operator": "AWS_SAGEMAKER",
        "sagemakerOperatorConfig": {
        "region": "us-east-1",
        "instanceType": "ml.m4.xlarge",
        "instanceCount": 1,
        "apiName": "predict",
        "timeout": 60
        }
    },
    "state": {
        "state": "RUNNING",
        "infoJson": {
        "EndpointName": "dev-bert-moviereview-sagemaker",
        "EndpointArn": "arn:aws:sagemaker:us-east-1:899399195124:endpoint/dev-sagemaker-moviereview-sagemaker",
        "EndpointConfigName": "dev-bert-moviereview-sagemaker-Service-20200702134432_033DAB",
        "ProductionVariants": [
            {
            "VariantName": "dev-sagemaker-moviereview-sagemaker-Service-20200702134432_033DAB",
            "DeployedImages": [
                {
                "SpecifiedImage": "899399195124.dkr.ecr.us-east-1.amazonaws.com/service-sagemaker:20200702134432_033DAB",
                "ResolvedImage": "899399195124.dkr.ecr.us-east-1.amazonaws.com/service-sagemaker@sha256:c064de18b75b18da26f5b8743491e13542a179915d5ea36ce4b8e971c6611062",
                "ResolutionTime": "2020-06-25 20:53:14.176000-04:00"
                }
            ],
            "CurrentWeight": 1.0,
            "DesiredWeight": 1.0,
            "CurrentInstanceCount": 1,
            "DesiredInstanceCount": 1
            }
        ],
        "EndpointStatus": "InService",
        "CreationTime": "2020-06-25 20:53:09.599000-04:00",
        "LastModifiedTime": "2020-06-25 20:59:33.149000-04:00",
        "ResponseMetadata": {
            "RequestId": "202c6fcf-048c-45e8-ab11-3dcc5771072b",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
            "x-amzn-requestid": "202c6fcf-048c-45e8-ab11-3dcc5771072b",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "831",
            "date": "Fri, 26 Jun 2020 00:59:34 GMT"
            },
            "RetryAttempts": 0
        }
        },
        "timestamp": "2020-06-26T00:59:34.850115Z"
    },
    "createdAt": "2020-06-26T00:15:56.839917Z",
    "lastUpdatedAt": "2020-06-26T00:15:56.839947Z"
    }


======================================
3.6 Test API Endpoints Using Boto3 SDK
======================================

Now, we are ready to test the SageMaker API endpoints by creating a small script using the AWS boto3 SDK. Alternatively, users could also use the AWS CLI to test the endpoint. Please visit `here <https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sagemaker-runtime/invoke-endpoint.html>`_

.. code-block:: python

    import boto3
    import json

    endpoint = 'dev-sagemaker-moviereview-deployment'
    runtime = boto3.Session().client('sagemaker-runtime')

    movie_example = '["The acting was a bit lacking."]'

    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/json', Body=movie_example)
    # Unpack response
    result = json.loads(response['Body'].read().decode())

    print(result)

    # sample output 

    ['negative']


4 Terminate AWS Resources
-------------------------

Lastly, do not forget to terminate the AWS resources used in this tutorial. Users could also clean up used resources by logging into the SageMaker console. For more information, please see `here <https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html>`_ 

.. code-block:: python

    bucket_to_delete = boto3.resource('s3').Bucket('movie-review-dataset')
    bucket_to_delete.objects.all().delete()
    sagemaker.Session().delete_endpoint('dev-sagemaker-moviereview-deployment')










