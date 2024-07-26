# Serving ARIMA model with BentoML 

This project shows how to apply a continuous learning ARIMA model
for time-series data in BentoML to forecasts future values.

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

Open in browser http://0.0.0.0:3000 to predict forecast.

```bash
curl -X 'POST' 'http://0.0.0.0:3000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[5]'
```

Sample result:
```
[
  21.32297249948254,
  39.103166807895505,
  51.62030696797619,
  57.742863144656305,
  57.316390331155915
]

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
