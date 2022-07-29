# LightGBM Example

This example trains a LightGBM classifier with the iris dataset and logs hyperparameters, metrics, and trained model.

It is modified from MLFlow LightGBM native example: https://github.com/mlflow/mlflow/tree/master/examples/lightgbm/


## Running the code

```
python train.py --colsample-bytree 0.8 --subsample 0.9
```
You can try experimenting with different parameter values like:
```
python train.py --learning-rate 0.4 --colsample-bytree 0.7 --subsample 0.8
```

Then you can open the MLflow UI to track the experiments and compare your runs via:
```
mlflow ui
```

## Running the code as a project

```
mlflow run . -P learning_rate=0.2 -P colsample_bytree=0.8 -P subsample=0.9
```

## Serve the model with BentoML

```
bentoml serve
```

```
curl -X POST -H "content-type: application/json" --data "[[5.9, 3, 5.1, 1.8]]" http://127.0.0.1:3000/classify
```

## Build Bento for production deployment

```
bentoml build
```

Generate docker image from Bento:
```
bentoml containerize lgb_iris_service:latest
```
