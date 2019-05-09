# Deploy BentoML service as serverless function on AWS Lambda or Google Cloud Function

## Overview

Many cloud providers offer serverless computing service to help teams deploy a
scalable services without worry about hardware configuration and maintains. This
benefit may also apply to machine learning as well.

In this example, we will train a text classification model with Tensorflow, and
then use BentoML to create serverless archive and deploy to AWS lambda service
with the Serverless framework.

## Prerequisites

1. Install Node.JS. Follow the instructions on [Nodejs.org](https://nodejs.org/en)
2. Install Serverless framework.  You can find instructions [here](https://serverless.com/framework/docs/getting-started/)
3. AWS account configured on your machine
   1. Install AWS CLI. [Instructions](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
   2. Configuring with your AWS account. [Instructions](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)


## Deploy to AWS lambda
It is simple to deploy to AWS Lambda with BentoML. After you saved your model as BentoML bundle, you invoke a single command.

```bash
bentoml deploy /ARCHIVE_PATH --platform aws-lambda --region us-west2
```

BentoML will create a deployment snapshot at your home directory's `.bentoml` directory.
BentoML then will use serverless framework to deploy to AWS lambda.

## Delete deployment from AWS lambda
Delete deployment from AWS lambda is as simple as deploy it. To delete deployment use `bentoml delete-deployment` command.
```bash
bentoml delete-deployment /ARCHIVE_PATH --platform aws-lambda --region us-west-2
```