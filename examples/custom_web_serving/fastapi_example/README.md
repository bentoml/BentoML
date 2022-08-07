# BentoML 🤝 FastAPI Demo Project

0. Install dependencies:

```bash
pip install -r ./requirements.txt
```

1. Train an Iris classifier model, similiar to the quickstart guide:

```bash
python ./train.py
```

2. Run the service:

```bash
bentoml serve service.py:svc
```

3. Send test request

Test the `/predict_bentoml` endpoint:

```bash
$ curl -X POST -H "content-type: application/json" --data '{"sepal_len": 7.2, "sepal_width": 3.2, "petal_len": 5.2, "petal_width": 2.2}' http://127.0.0.1:3000/predict_bentoml

[2]%
```

Now for FastAPI integration endpoints:

Test the `/predict_fastapi` endpoint:

```bash
$ curl -X POST -H "content-type: application/json" --data '{"sepal_len": 6.2, "sepal_width": 3.2, "petal_len": 5.2, "petal_width": 2.2}' http://127.0.0.1:3000/predict_fastapi

{"prediction":2}%
```

Test the `/predict_fastapi_async` endpoint:

```bash
$ curl -X POST -H "content-type: application/json" --data '{"sepal_len": 6.2, "sepal_width": 3.2, "petal_len": 5.2, "petal_width": 2.2}' http://127.0.0.1:3000/predict_fastapi

{"prediction":2}%
```

Test the custom `/metadata` endpoint

```bash
$ curl http://127.0.0.1:3000/metadata

{"name":"iris_clf_with_feature_names","version":"kgjn3haltwvixuqj"}%
```


4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize iris_fastapi_demo:latest
```
