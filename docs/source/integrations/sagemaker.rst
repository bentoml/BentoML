=============
AWS SageMaker
=============

BentoML makes deploying models to SageMaker easier! Try deploying your model with
BentoML and bentoctl to SageMaker: https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html
Model deployment on SageMaker natively requires users to build their API server with
Flask/FastAPI and containerize the flask app by themselves, when not using the built-in
algorithms. BentoML provides a high-performance API server for users without the need
for lower-level web server development.
