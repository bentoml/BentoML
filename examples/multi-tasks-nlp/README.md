# BentoML Multi-task NLP Quickstart

This is a quickstart project demonstrating basic usage of
[BentoML](https://github.com/bentoml) with Transformers.

In this project, we will use
[Meta's BART](https://huggingface.co/sshleifer/distilbart-cnn-12-6) model for
summarization and zero-shot classification (categorization), and
[a BERT variant](https://huggingface.co/ProsusAI/finbert) for sentiment
analysis. We will build a prediction service for serving the pre-trained model,
and containerize the model server as a container for production.

### Install Dependencies

Install python packages required for running this project:

```bash
pip install -r ./requirements.txt
```

### Model Warmup

Firstly, we will run a warmup step to save all required models to BentoML model
store, via `warmup.py` script:

```bash
python3 warmup.py
```

This will save required models in the BentoML local model store, a new version
tag is automatically generated when the model is saved. You can see all model
revisions from CLI via `bentoml models` commands:

```bash
bentoml models get summarizer-pipeline:latest

bentoml models get categorizer-pipeline:latest

bentoml models get sentimenter-pipeline:latest

bentoml models list

bentoml models --help
```

### Serving the model

Define the following BentoML service in a `service.py` file:

```python
import bentoml

summarizer = bentoml.transformers.get("summarizer-pipeline").to_runner()
categorizer = bentoml.transformers.get("categorizer-pipeline").to_runner()
sentimenter = bentoml.transformers.get("sentimenter-pipeline").to_runner()

svc = bentoml.Service(
    name="multi-task-nlp", runners=[summarizer, categorizer, sentimenter]
)


@svc.api(input=bentoml.io.Text.from_sample(TEXT), output=bentoml.io.Text())
async def summarize(text: str) -> str:
    generated = await summarizer.async_run(text, max_length=MAX_LENGTH)
    return generated[0]["summary_text"]

@svc.api(
    input=bentoml.io.JSON.from_sample({"text": TEXT, "categories": CATEGORIES}),
    output=bentoml.io.JSON(),
)
async def categorize(
    input_data: dict[t.Literal["text", "categories"], str | list[str]]
) -> dict[str, float]:
    predictions = await categorizer.async_run(
        input_data["text"], input_data["categories"], multi_label=True
    )
    return {
        c: p
        for c, p in zip(predictions["labels"], predictions["scores"])
    }


@svc.api(input=bentoml.io.Text.from_sample(TEXT), output=bentoml.io.JSON())
async def sentiment_analysis(text: str) -> dict[str, float]:
    predictions = await sentimenter.async_run(text)
    return {c["label"]: c["score"] for c in predictions}
```

Let's unpack the above code:

1. We create three
   [BentoML's Runner](https://docs.bentoml.org/en/latest/concepts/runner.html)
   from the saved models, and create a new service named `multi-task-nlp`.
2. We define three APIs for the service, and runs the runners corespondingly for
   each of the tasks (summarization, categorization and sentiment analysis).

Run the service locally with `bentoml serve-http` command:

```bash
bentoml serve-http service.py:svc
```

Open your web browser at http://127.0.0.1:3000 to view the Bento UI for sending
test requests.

> NOTE: You can also serve with gRPC with `bentoml serve-grpc` command. Make
> sure to install bentoml with `pip install bentoml[grpc]` to use gRPC:
>
> ```bash
> bentoml serve-grpc service.py:svc
> ```
>
> You can use tools such as [grpcui](https://github.com/fullstorydev/grpcui) or
> [grpcurl](https://github.com/fullstorydev/grpcurl) to send requests to the
> running gRPC server.

### Build Bento for deployment

Bento is the distribution format in BentoML which captures all the source code,
model files, config files and dependency specifications required for running the
service for production deployment. Think of it as Docker/Container designed for
machine learning models.

To begin with building Bento, create a `bentofile.yaml` under your project
directory:

```yaml
service: "service.py:svc"
labels:
  owner: bentoml-team
  project: multi-tasks-nlp
include:
  - "*.py"
python:
  requirements_txt: ./requirements.txt
```

Next, run `bentoml build` from current directory to start the Bento build:

```bash
bentoml build

# ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
# ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
# ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
# ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
# ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
# ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝
#
# Successfully built Bento(tag="multi-tasks-nlp:i3l36cwffs553gxi").
```

A new Bento is now built and saved to local Bento store. You can view and manage
it via `bentoml list`,`bentoml get` and `bentoml delete` CLI command.

### Containerize and Deployment

Bento is designed to be deployed to run efficiently in a variety of different
environments. And there are lots of deployment options and tools as part of the
BentoML eco-system, such as [Yatai](https://github.com/bentoml/Yatai) and
[bentoctl](https://github.com/bentoml/bentoctl) for direct deployment to cloud
platforms.

In this guide, we will show you the most basic way of deploying a Bento, which
is converting a Bento into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the
following commnand:

```bash
bentoml containerize multi-tasks-nlp:i3l36cwffs553gxi --opt platform=linux/amd64
```

This will build a new docker image with all source code, model files and
dependencies in place, and ready for production deployment. To start a container
with this docker image locally, run:

```bash
docker run -p 3000:3000 multi-tasks-nlp:i3l36cwffs553gxi serve --production
```

## What's Next?

- 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy
  to help with any issue you face or even just to meet you and hear what you're
  working on.
- Dive deeper into the
  [Core Concepts](https://docs.bentoml.org/en/latest/concepts/index.html) in
  BentoML
- Learn how to use BentoML with other ML Frameworks at
  [Frameworks Guide](https://docs.bentoml.org/en/latest/frameworks/index.html)
  or check out other
  [gallery projects](https://github.com/bentoml/BentoML/tree/main/examples)
- Learn more about model deployment options for Bento:
  - [🦄️ Yatai](https://github.com/bentoml/Yatai): Model Deployment at scale on
    Kubernetes
  - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on
    any cloud platform
