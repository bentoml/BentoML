##  Use BentoML with AWS Lambda

In this exmaple, we will create a simple machine learning model, use BentoML to export the model and its dependencies, and deploy the model as an endpoint to AWS lambda.  

#### Directory Structure
```
/model
- notebook
- something
/serverless
- handler.py
- serverless.yml
```