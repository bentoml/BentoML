# Deploy BentoML service with Serverless

## Overview

Many cloud providers offer serverless computing service to help teams deploy a
scalable services without worry about hardware configuration and maintains. This
benefit may also apply to machine learning as well.

In this example, we will train a text classification model with Tensorflow, and
then use BentoML to create serverless archive and deploy to AWS lambda service
with the Serverless framework.

## Prerequisites

1. Install Serverless framework.  You can find instructions [here](https://serverless.com/framework/docs/getting-started/)
2. Have AWS account setup.


## Workflow
There are few steps to get model deployed to AWS lambda after finish training
model.

1. Write BentoService configuration that include the artifacts you want to pack
   and write the prediction function.
2. Save the trained model to filesystem with the BentoService class you just created, 
3. Call `bentoml build-serverless-archive` command to generate serverless
   project based on the saved BentoService
4. [Optional] Update serverless configuration that suit for your deployment
   needs
5. Deploy to AWS lambda with 'serverless deploy' command


## Anatomy of bentoml build-serverless-archive command
The command is building serverless project base on the already archived bento
service.  It will update serverless configuration with BentoService info, also
generate handler file that use your bento service to predict.

The commands take two arguments, an path to the BentoService that you want to
generate serverless project from, and a directory path for the serverless
project to be saved at.

You could also pass in `stage` and `region` as options


## Anatomy of generated serverless project
The generated servereless project includes:

* your bento service archive. It is a copy of the archive directory you saved.
* requirements.txt.  Your bento service dependencies are copied to the top level
  for 'serverless-python-requirements' plugin to consume.
* an updated servereless configuration.  Preinstalled with plugin
  `serverless-python-requirements`, updated functions section.
* handler.py.  The generated handler.py file setup to do inferencing with the
  Bento service.  It will work out of the box.


## Conclusion

As you can see, train a model and deploy to a serverless service is very easy
with BentoML.  You can generate a complete serverless project with a single
command.
