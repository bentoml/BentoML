# BentoML Inference Graph Tutorial

This is a sample project demonstrating model inference graph of [BentoML](https://github.com/bentoml)
with Huggingface Transformers.

In this project, we will download and save three pretrained text generation models and a pretrained text classification model
to the model store. We will then build a service that accepts a text input, passes the input to the three text generation models,
classify each generated paragraph with the classification model, and return all three generated paragraphs with their classification
scores. The service will be served via HTTP and containerized as a docker image for production deployment.

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements.txt
```

### Model Training

Create and save three text generation models and one text classification model.

```bash
import bentoml
import transformers


if __name__ == "__main__":
    # Create Transformers pipelines from pretrained models
    generation_pipeline_1 = transformers.pipeline(
        task="text-generation",
        model="gpt2",
    )
    generation_pipeline_2 = transformers.pipeline(
        task="text-generation",
        model="distilgpt2",
    )
    generation_pipeline_2 = transformers.pipeline(
        task="text-generation",
        model="gpt2-medium",
    )

    classification_pipeline = transformers.pipeline(
        task="text-classification",
        model="bert-base-uncased",
        tokenizer="bert-base-uncased",
    )

    # Save models to BentoML local model store
    m0 = bentoml.transformers.save_model("gpt2-generation", generation_pipeline_1)
    m1 = bentoml.transformers.save_model("distilgpt2-generation", generation_pipeline_2)
    m2 = bentoml.transformers.save_model(
        "gpt2-medium-generation", generation_pipeline_2
    )
    m3 = bentoml.transformers.save_model(
        "bert-base-uncased-classification", classification_pipeline
    )
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

pipeline = bentoml.transformers.load_model("gpt2-generation:latest")

pipeline("I have an idea!")
```

In BentoML, the recommended way of running ML model inference in serving is via Runners, which
gives BentoML more flexibility in scheduling the inference computation, batching inference requests,
and taking advantage of hardware resoureces available. Saved models can be loaded as Runner instance as
shown below:

```python
import bentoml

# Create a Runner instance:
bert_runner = bentoml.transformers.get("gpt2-generation:latest").to_runner()

# Runner#init_local initializes the model in current process, this is meant for development and testing only:
bert_runner.init_local()

# This should yield the same result as the loaded model:
bert_runner.run("I have an idea!")
```


### Serving the model

The service definition below achieves the inference graph logic described above.

First, the we create three text generation runners and one text classification runners with the `to_runner` function
from the models we previously saved. Second, we create a `bentoml.Service` named "inference_graph" with pass in
the four runners instances. Lastly, we create an async `@svc.api` that accepts a `Text` input and `JSON` output. The API
passes the input simultaneously to all three text generation models through `asyncio.gather` and iteratively passes
the generated paragraphs to the text classification model. The API returns all three generated paragraphs and their
corresponding classification scores as a dictionary.

```python
import asyncio

import bentoml
from bentoml.io import JSON
from bentoml.io import Text

gpt2_generator = bentoml.transformers.get("gpt2-generation:latest").to_runner()
distilgpt2_generator = bentoml.transformers.get(
    "distilgpt2-generation:latest"
).to_runner()
distilbegpt2_medium_generator = bentoml.transformers.get(
    "gpt2-medium-generation:latest"
).to_runner()
bert_base_uncased_classifier = bentoml.transformers.get(
    "bert-base-uncased-classification:latest"
).to_runner()

svc = bentoml.Service(
    "inference_graph",
    runners=[
        gpt2_generator,
        distilgpt2_generator,
        distilbegpt2_medium_generator,
        bert_base_uncased_classifier,
    ],
)


MAX_LENGTH = 128
NUM_RETURN_SEQUENCE = 1


@svc.api(input=Text(), output=JSON())
async def classify_generated_texts(original_sentence: str) -> dict:
    generated_sentences = [
        result[0]["generated_text"]
        for result in await asyncio.gather(
            gpt2_generator.async_run(
                original_sentence,
                max_length=MAX_LENGTH,
                num_return_sequences=NUM_RETURN_SEQUENCE,
            ),
            distilgpt2_generator.async_run(
                original_sentence,
                max_length=MAX_LENGTH,
                num_return_sequences=NUM_RETURN_SEQUENCE,
            ),
            distilbegpt2_medium_generator.async_run(
                original_sentence,
                max_length=MAX_LENGTH,
                num_return_sequences=NUM_RETURN_SEQUENCE,
            ),
        )
    ]

    results = []
    for sentence in generated_sentences:
        score = (await bert_base_uncased_classifier.async_run(sentence))[0]["score"]
        results.append(
            {
                "generated": sentence,
                "score": score,
            }
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
  'http://0.0.0.0:3000/classify_generated_texts' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -d 'I have an idea!'
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
    - torch
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

- 👉 [Pop into our Slack community!](https://l.bentoml.com/join-slack) We're happy to help with any issue you face or even just to meet you and hear what you're working on.
- Dive deeper into the [Core Concepts](https://docs.bentoml.org/en/latest/concepts/index.html) in BentoML
- Learn how to use BentoML with other ML Frameworks at [Frameworks Guide](https://docs.bentoml.org/en/latest/frameworks/index.html) or check out other [gallery projects](https://github.com/bentoml/BentoML/tree/main/examples)
- Learn more about model deployment options for Bento:
  - [🦄️ Yatai](https://github.com/bentoml/Yatai): Model Deployment at scale on Kubernetes
  - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on any cloud platform
