# Serving ARIMA model with BentoML 

This project demonstrates how to use a continuous learning ARIMA model 
for a time-series data and use it as a prediction service in BentoML.

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

Open in browser http://0.0.0.0:3000 to submit input images via the web UI.

```bash
curl -X 'POST' 'http://0.0.0.0:3000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[20]'
```

Sample result:
```
14.584079431935686
```

## Build Bento

Build Bento using the bentofile.yaml which contains all the configurations required:

```bash
bentoml build -f ./bentofile.yaml
```

Once the Bento is built, containerize it as a Docker image for deployment:

```bash
bentoml containerize arima_forecast_model:latest
```
