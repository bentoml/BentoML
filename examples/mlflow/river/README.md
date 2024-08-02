# Serving a model from river with BentoML 

This project shows how to train a model using river online machine learning library [river](https://riverml.xyz/latest/)
and log the model using MLflow's custom python function 
[``mlflow.pyfunc.log_model``](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model)
and import the model in BentoML model store for model serving.

## Requirements

Install requirements with:

```bash
pip install -r ./requirements.txt
```

## Instruction

1. Train and save model:

```bash
python ./train.py
```

2. Run the service:

```bash
bentoml serve service.py:svc
```

## Test the endpoint

Open in browser http://0.0.0.0:3000 to predict a value.

```bash
curl -X 'POST' 'http://0.0.0.0:3000/predict' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
        "ordinal_date": 836489,
        "gallup": 47.843213,
        "ipsos": 48.07067899999999,
        "morning_consult": 52.318749,
        "rasmussen": 50.104692,
        "you_gov": 58.636914000000004
        }'
```

Sample result:
```
42.74910074533503
```

## Build Bento

Build Bento using the bentofile.yaml which contains all the configurations required:

```bash
bentoml build -f ./bentofile.yaml
```

Once the Bento is built, containerize it as a Docker image for deployment:

```bash
bentoml containerize river_arf_model:latest
```