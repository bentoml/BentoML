# BentoML 🤝 Flask Demo Project

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
$ curl -X POST -H "content-type: application/json" --data "[[5.9, 3, 5.1, 1.8]]" http://127.0.0.1:3000/predict_bentoml

[2]%
```

Now for Flask integration endpoints:

Test the custom `/metadata` endpoint:

```bash
$ curl http://127.0.0.1:3000/metadata

{"name":"iris_clf","version":"3vl5n7qkcwqe5uqj"}
```

Test the custom `/predict_flask` endpoint:

```bash
$ curl -X POST -H "content-type: application/json" --data "[[5.9, 3, 5.1, 1.8]]" http://127.0.0.1:3000/predict_flask

[2]
```

4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize iris_flask_demo:latest
```
