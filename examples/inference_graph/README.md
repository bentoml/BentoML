# BentoML Inference Graph Tutorial

This is a sample project demonstrating model inference graph of [BentoML](https://github.com/bentoml) 
with Huggingface Transformers.

In this project, we will download three pretrained models, save them as three text classification
Transformers pipelines, build a text classification service via an HTTP server, and containerize the 
service as a docker image for production deployment.

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements.txt
```

### Model Training

First step, create and save three text classification pipelines from three different BERT models:

```bash
import bentoml
import transformers


if __name__ == "__main__":
    # Create Transformers pipelines from pretrained models
    pipeline1 = transformers.pipeline(task="text-classification", model="bert-base-uncased", tokenizer="bert-base-uncased")
    pipeline2 = transformers.pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    pipeline3 = transformers.pipeline(task="text-classification", model="ProsusAI/finbert")

    # Save models to BentoML local model store
    bentoml.transformers.save_model("bert-base-uncased", pipeline1)
    bentoml.transformers.save_model("distilbert-base-uncased-finetuned-sst-2-english", pipeline2)
    bentoml.transformers.save_model("prosusai-finbert", pipeline3)

```

This will save the models in the BentoML local model store, new version tags are automatically
generated when the models are saved. You can see all model revisions from CLI via `bentoml models`
commands:

```bash
bentoml models list

bentoml models --help
```

To verify that the saved model can be loaded correctly, run the following:

```python
import bentoml

pipeline = bentoml.transformers.load_model("bert-base-uncased:latest")

pipeline("You look great today!")
```

In BentoML, the recommended way of running ML model inference in serving is via Runner, which 
gives BentoML more flexibility in scheduling the inference computation, batching inference requests, 
and taking advantage of hardware resoureces available. Saved models can be loaded as Runner instance as 
shown below:

```python
import bentoml

# Create a Runner instance:
bert_runner = bentoml.transformers.get("bert-base-uncased:latest").to_runner()

# Runner#init_local initializes the model in current process, this is meant for development and testing only:
bert_runner.init_local()

# This should yield the same result as the loaded model:
bert_runner.run("You look great today!")
```


### Serving the model

A simple BentoML Service that serves the model saved above look like this:

```python
import asyncio
import bentoml

from bentoml.io import Text, JSON
from statistics import median

bert_runner = bentoml.transformers.get("bert-base-uncased:latest").to_runner()
distilbert_runner = bentoml.transformers.get("distilbert-base-uncased-finetuned-sst-2-english:latest").to_runner()
finbert_runner = bentoml.transformers.get("prosusai-finbert:latest").to_runner()

svc = bentoml.Service("inference_graph", runners=[bert_runner, distilbert_runner, finbert_runner])

@svc.api(input=Text(), output=JSON())
async def classify(input_data: str) -> dict:
    results = await asyncio.gather(
        bert_runner.async_run(input_data),
        distilbert_runner.async_run(input_data),
        finbert_runner.async_run(input_data),
    )
    return results
```

```bash
bentoml serve --reload
```

Open your web browser at http://127.0.0.1:3000 to view the Bento UI for sending test requests.

You may also send request with `curl` command or any HTTP client, e.g.:

```bash
curl -X 'POST' \
  'http://127.0.0.1:3000/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d 'You look great today!'
```


### Build Bento for deployment

Bento is the distribution format in BentoML which captures all the source code, model files, config
files and dependency specifications required for running the service for production deployment. Think 
of it as Docker/Container designed for machine learning models.

To begin with building Bento, create a `bentofile.yaml` under your project directory:

```yaml
service: "service.py:svc"
labels:
  owner: bentoml-team
  project: gallery
include:
- "*.py"
python:
  packages:
    - transformers
```

Next, run `bentoml build` from current directory to start the Bento build:

```
> bentoml build

Jax version 0.2.19, Flax version 0.3.4 available.
Building BentoML service "inference_graph:owljo4hna25nblg6" from build context "/Users/ssheng/github/gallery/inference_graph"
Packing model "prosusai-finbert:pomvfgxm7kh4rlg6"
Successfully saved Model(tag="prosusai-finbert:pomvfgxm7kh4rlg6")
Packing model "distilbert-base-uncased-finetuned-sst-2-english:pm7gbexm7kh4rlg6"
Successfully saved Model(tag="distilbert-base-uncased-finetuned-sst-2-english:pm7gbexm7kh4rlg6")
Packing model "bert-base-uncased:pla6cshm7kh4rlg6"
Successfully saved Model(tag="bert-base-uncased:pla6cshm7kh4rlg6")
Locking PyPI package versions..

██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

Successfully built Bento(tag="inference_graph:owljo4hna25nblg6")
```

A new Bento is now built and saved to local Bento store. You can view and manage it via 
`bentoml list`,`bentoml get` and `bentoml delete` CLI command.


### Containerize and Deployment

Bento is designed to be deployed to run efficiently in a variety of different environments.
And there are lots of deployment options and tools as part of the BentoML eco-system, such as 
[Yatai](https://github.com/bentoml/Yatai) and [bentoctl](https://github.com/bentoml/bentoctl) for
direct deployment to cloud platforms.

In this guide, we will show you the most basic way of deploying a Bento, which is converting a Bento
into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the following commnand:

```bash
bentoml containerize inference_graph:latest
```

This will build a new docker image with all source code, model files and dependencies in place,
and ready for production deployment. To start a container with this docker image locally, run:

```bash
docker run -p 3000:3000 inference_graph:invwzzsw7li6zckb2ie5eubhd 
```

## What's Next?

- 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy to help with any issue you face or even just to meet you and hear what you're working on.
- Dive deeper into the [Core Concepts](https://docs.bentoml.org/en/latest/concepts/index.html) in BentoML
- Learn how to use BentoML with other ML Frameworks at [Frameworks Guide](https://docs.bentoml.org/en/latest/frameworks/index.html) or check out other [gallery projects](https://github.com/bentoml/BentoML/tree/main/examples)
- Learn more about model deployment options for Bento:
  - [🦄️ Yatai](https://github.com/bentoml/Yatai): Model Deployment at scale on Kubernetes
  - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on any cloud platform

