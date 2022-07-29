# Custom LDA classifier model via bentoml.picklable_model

This example is based on https://github.com/eriklindernoren/ML-From-Scratch

`bentoml.picklable_model` represents a generic model type in BentoML, that uses
`cloudpickle` for model serialization under the hood. Most pure python code based
ML model implementation should work with `bentoml.picklable_model` out-of-the-box.

0. Install dependencies:

```bash
pip install -r ./requirements.txt
```

1. Train a custom LDA model:

```bash
python ./train.py
```

2. Run the service:

```bash
bentoml serve service.py:svc
```

3. Send test request

```
curl -X POST -H "content-type: application/json" --data "[[5.9, 3, 5.1, 1.8]]" http://127.0.0.1:3000/classify
```

4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize iris_classifier_lda:latest
```
