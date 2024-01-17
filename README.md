<div align="center">
  <img src="https://github.com/bentoml/BentoML/assets/489344/398274c1-a572-477b-b115-52497a085496" width="180px" alt="bentoml" />
  <h1 align="center">BentoML: The Unified AI Application Framework</h1>
  <a href="https://pypi.org/project/bentoml"><img src="https://img.shields.io/pypi/v/bentoml.svg" alt="pypi_status" /></a>
  <a href="https://github.com/bentoml/BentoML/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/bentoml/bentoml/workflows/CI/badge.svg?branch=main" alt="CI" /></a>
  <a href="https://twitter.com/bentomlai"><img src="https://badgen.net/badge/icon/@bentomlai/1DA1F2?icon=twitter&label=Follow%20Us" alt="Twitter" /></a>
  <a href="https://join.slack.bentoml.org"><img src="https://badgen.net/badge/Join/Community/cyan?icon=slack" alt="Community" /></a>
  <p>BentoML is a framework for building <b>reliable, scalable, and cost-efficient AI
applications</b>. It comes with everything you need for model serving, application
packaging, and production deployment.</p>
  <i><a href="https://l.bentoml.com/join-slack">👉 Join our Slack community!</a></i>
</div>

# Highlights

### 🍱 Bento is the container for AI apps

- Open standard and SDK for AI apps, pack your code, inference pipelines, model
  files, dependencies, and runtime configurations in a
  [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html).
- Auto-generate API servers, supporting REST API, gRPC, and long-running
  inference jobs.
- Auto-generate Docker container images.

### 🏄 Freedom to build with any AI models

- Import from any model hub or bring your own models built with frameworks like
  PyTorch, TensorFlow, Keras, Scikit-Learn, XGBoost and many more.
- Native support for
  [LLM inference](https://github.com/bentoml/openllm/#bentoml),
  [generative AI](https://github.com/bentoml/stable-diffusion-bentoml),
  [embedding creation](https://github.com/bentoml/CLIP-API-service), and
  [multi-modal AI apps](https://github.com/bentoml/Distributed-Visual-ChatGPT).
- Run and debug your BentoML apps locally on Mac, Windows, or Linux.

### 🤖️ Inference optimization for AI applications

- Integrate with high-performance runtimes such as ONNX-runtime and TorchScript to boost response time and throughput.
- Support parallel processing of model inferences for improved speed and efficiency.
- Implement adaptive batching to optimize processing.
- Built-in optimization for specific model architectures (like OpenLLM for LLMs).

### 🍭 Simplify modern AI application architecture

- Python-first! Effortlessly scale complex AI workloads.
- Enable GPU inference without the headache.
- Compose multiple models to run concurrently or sequentially, over multiple GPUs or
  [on a Kubernetes Cluster](https://github.com/bentoml/yatai).
- Natively integrates with MLFlow, [LangChain](https://github.com/ssheng/BentoChain),
  Kubeflow, Triton, Spark, Ray, and many more to complete your production AI stack.

### 🚀 Deploy anywhere

- One-click deployment to [☁️ BentoCloud](https://bentoml.com/cloud), the
  Serverless platform made for hosting and operating AI apps.
- Scalable BentoML deployment with [🦄️ Yatai](https://github.com/bentoml/yatai)
  on Kubernetes.
- Deploy auto-generated container images anywhere Docker runs.

# Documentation

- Installation: `pip install "bentoml>=1.2.0a0"`
- Documentation: [docs.bentoml.com](https://docs.bentoml.com/en/latest/)
- Tutorial: [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html)

### 🛠️ What you can build with BentoML

- [OpenLLM](https://github.com/bentoml/OpenLLM) - An open platform for operating
  large language models (LLMs) in production.
- [StableDiffusion](https://github.com/bentoml/stable-diffusion-bentoml) -
  Create your own image generation service with any diffusion models..
- [CLIP-API-service](https://github.com/bentoml/CLIP-API-service) - Embed images
  and sentences, object recognition, visual reasoning, image classification, and
  reverse image search.
- [Transformer NLP Service](https://github.com/bentoml/transformers-nlp-service) -
  Online inference API for Transformer NLP models.
- [Distributed TaskMatrix(Visual ChatGPT)](https://github.com/bentoml/Distributed-Visual-ChatGPT) -
  Scalable deployment for TaskMatrix from Microsoft.
- [Fraud Detection](https://github.com/bentoml/Fraud-Detection-Model-Serving) -
  Online model serving with custom XGBoost model.
- [OCR as a Service](https://github.com/bentoml/OCR-as-a-Service) - Turn any OCR
  models into online inference API endpoints.
- [Replace Anything](https://github.com/yuqwu/Replace-Anything) - Combine the
  power of Segment Anything and Stable Diffusion.
- [DeepFloyd IF Multi-GPU serving](https://github.com/bentoml/IF-multi-GPUs-demo) -
  Serve IF models easily across multiple GPUs.
- [Sentence Embedding as a Service](https://github.com/bentoml/sentence-embedding-bento) -
  Start a high-performance REST API server for generating text embeddings with one command.
- Check out more examples
  [here](https://github.com/bentoml/BentoML/tree/main/examples).

# Getting started

This example demonstrates how to serve and deploy a simple text summarization application.

## Serving a model locally

Install dependencies:

```
pip install torch transformers "bentoml>=1.2.0a0"
```

Define the serving logic of your model in a `service.py` file.

```python
from __future__ import annotations
import bentoml
from transformers import pipeline


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class Summarization:
    def __init__(self) -> None:
        # Load model into pipeline
        self.pipeline = pipeline('summarization')

    @bentoml.api
    def summarize(self, text: str) -> str:
        result = self.pipeline(text)
        return result[0]['summary_text']
```

Run this BentoML Service locally, which is accessible at [http://localhost:3000](http://localhost:3000).

```bash
bentoml serve service:Summarization
```

Send a request to summarize a short news paragraph:

```bash
curl -X 'POST' \
  'http://localhost:3000/summarize' \
  -H 'accept: text/plain' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson'\''s cat, Whiskers, performed what witnesses are calling a '\''miraculous and gravity-defying leap.'\'' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson'\''s backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as '\''The Leap of the Century."
}'
```

## Deployment

After your Service is ready, you can deploy it to BentoCloud or as a Docker image.

First, create a `bentofile.yaml` file for building a Bento.

```yaml
service: "service:Summarization"
labels:
  owner: bentoml-team
  project: gallery
include:
  - "*.py"
python:
  packages:
  - torch
  - transformers
```

Then, choose one of the following ways for deployment:

<details>

<summary>BentoCloud</summary>

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html) and then run the following command:

```bash
bentoml deploy .
```

</details>

<details>

<summary>Docker</summary>

Build a Bento to package necessary dependencies and components into a standard distribution format.

```
bentoml build
```

Containerize the Bento.

```
bentoml containerize summarization:latest
```

Run this image with Docker.

```bash
docker run -p 3000:3000 summarization:latest
```

</details>

For detailed explanations, read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html).

---

## Community

BentoML supports billions of model runs per day and is used by thousands of
organizations around the globe.

Join our [Community Slack 💬](https://l.bentoml.com/join-slack), where thousands
of AI application developers contribute to the project and help each other.

To report a bug or suggest a feature request, use
[GitHub Issues](https://github.com/bentoml/BentoML/issues/new/choose).

## Contributing

There are many ways to contribute to the project:

- Report bugs and "Thumbs up" on issues that are relevant to you.
- Investigate issues and review other developers' pull requests.
- Contribute code or documentation to the project by submitting a GitHub pull
  request.
- Check out the
  [Contributing Guide](https://github.com/bentoml/BentoML/blob/main/CONTRIBUTING.md)
  and
  [Development Guide](https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md)
  to learn more
- Share your feedback and discuss roadmap plans in the `#bentoml-contributors`
  channel [here](https://l.bentoml.com/join-slack).

Thanks to all of our amazing contributors!

<a href="https://github.com/bentoml/BentoML/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bentoml/BentoML" />
</a>

---

### Usage Reporting

BentoML collects usage data that helps our team to improve the product. Only
BentoML's internal API calls are being reported. We strip out as much
potentially sensitive information as possible, and we will never collect user
code, model data, model names, or stack traces. Here's the
[code](./src/bentoml/_internal/utils/analytics/usage_stats.py) for usage
tracking. You can opt-out of usage tracking by the `--do-not-track` CLI option:

```bash
bentoml [command] --do-not-track
```

Or by setting environment variable `BENTOML_DO_NOT_TRACK=True`:

```bash
export BENTOML_DO_NOT_TRACK=True
```

---

### License

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/main/LICENSE)

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=small)](https://app.fossa.com/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_small)

### Citation

If you use BentoML in your research, please cite using the following
[citation](./CITATION.cff):

```bibtex
@software{Yang_BentoML_The_framework,
author = {Yang, Chaoyu and Sheng, Sean and Pham, Aaron and  Zhao, Shenyang and Lee, Sauyon and Jiang, Bo and Dong, Fog and Guan, Xipeng and Ming, Frost},
license = {Apache-2.0},
title = {{BentoML: The framework for building reliable, scalable and cost-efficient AI application}},
url = {https://github.com/bentoml/bentoml}
}
```
