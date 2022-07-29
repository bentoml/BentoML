# Custom Runner with pre-trained NLTK model

For ML libraries that provide built-in trained models, such as NLTK, users may create a
custom Runner directly without saving the model to model store.

0. Install dependencies:

```bash
pip install -r ./requirements.txt
```

1. Download required NLTK trained models

```bash
./download_nldk_models.py
```

2. Run the service:

```bash
bentoml serve service.py:svc
```

3. Send test request

```
curl -X POST -H "content-type: application/text" --data "BentoML is great" http://127.0.0.1:3000/analysis
```

4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize sentiment_analyzer:latest
```

