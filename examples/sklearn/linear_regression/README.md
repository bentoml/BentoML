# BentoML Sklearn Example: Linear Regression

0. Install dependencies:

```bash
pip install -r ./requirements.txt
```

1. Train a linear regression model

```bash
python ./train.py
```

2. Run the service:

```bash
bentoml serve service.py:svc
```

3. Send test request

```
curl -X POST -H "content-type: application/json" --data "[[5, 3]]" http://127.0.0.1:3000/predict
```

4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize linear_regression:latest
```


