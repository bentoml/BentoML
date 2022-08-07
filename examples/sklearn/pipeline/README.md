# BentoML Sklearn Example: document classification pipeline

0. Install dependencies:

```bash
pip install -r ./requirements.txt
```

1. Train a document classification pipeline model

```bash
python ./train.py
```

2. Run the service:

```bash
bentoml serve service.py:svc
```

3. Send test request

Test the `/predict` endpoint:
```bash
curl -X POST -H "content-type: application/text" --data "hello world" http://127.0.0.1:3000/predict
```

Test the `/predict_proba` endpoint:
```bash
curl -X POST -H "content-type: application/text" --data "hello world" http://127.0.0.1:3000/predict_proba
```


4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize doc_classifier:latest
```


