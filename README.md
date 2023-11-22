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
  PyTorch, TensorFlow, Keras, Scikit-Learn, XGBoost and
  [many more](https://docs.bentoml.com/en/latest/frameworks/index.html).
- Native support for
  [LLM inference](https://github.com/bentoml/openllm/#bentoml),
  [generative AI](https://github.com/bentoml/stable-diffusion-bentoml),
  [embedding creation](https://github.com/bentoml/CLIP-API-service), and
  [multi-modal AI apps](https://github.com/bentoml/Distributed-Visual-ChatGPT).
- Run and debug your BentoML apps locally on Mac, Windows, or Linux.

### 🍭 Simplify modern AI application architecture

- Python-first! Effortlessly scale complex AI workloads.
- Enable GPU inference
  [without the headache](https://docs.bentoml.com/en/latest/guides/gpu.html).
- [Compose multiple models](https://docs.bentoml.com/en/latest/guides/graph.html)
  to run concurrently or sequentially, over
  [multiple GPUs](https://docs.bentoml.com/en/latest/guides/scheduling.html) or
  [on a Kubernetes Cluster](https://github.com/bentoml/yatai).
- Natively integrates with
  [MLFlow](https://docs.bentoml.com/en/latest/integrations/mlflow.html),
  [LangChain](https://github.com/ssheng/BentoChain),
  [Kubeflow](https://www.kubeflow.org/docs/external-add-ons/serving/bentoml/),
  [Triton](https://docs.bentoml.com/en/latest/integrations/triton.html),
  [Spark](https://docs.bentoml.com/en/latest/integrations/spark.html),
  [Ray](https://docs.bentoml.com/en/latest/integrations/ray.html), and many more
  to complete your production AI stack.

### 🚀 Deploy Anywhere

- One-click deployment to [☁️ BentoCloud](https://bentoml.com/cloud), the
  Serverless platform made for hosting and operating AI apps.
- Scalable BentoML deployment with [🦄️ Yatai](https://github.com/bentoml/yatai)
  on Kubernetes.
- Deploy auto-generated container images anywhere docker runs.

# Documentation

- Installation: `pip install bentoml`
- Full Documentation: [docs.bentoml.com](https://docs.bentoml.com/en/latest/)
- Tutorial: [Intro to BentoML](https://docs.bentoml.com/en/latest/tutorial.html)

### 🛠️ What you can build with BentoML

- [OpenLLM](https://github.com/bentoml/OpenLLM) - An open platform for operating
  large language models (LLMs) in production.
- [StableDiffusion](https://github.com/bentoml/stable-diffusion-bentoml) -
  Create your own text-to-image service with any diffusion models.
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

# Getting Started

Save or import models in BentoML local model store:

```python
import bentoml
import transformers

pipe = transformers.pipeline("text-classification")

bentoml.transformers.save_model(
  "text-classification-pipe",
  pipe,
  signatures={
    "__call__": {"batchable": True}  # Enable dynamic batching for model
  }
)
```

View all models saved locally:

```bash
$ bentoml models list

Tag                                     Module                Size        Creation Time
text-classification-pipe:kn6mr3aubcuf…  bentoml.transformers  256.35 MiB  2023-05-17 14:36:25
```

Define how your model runs in a `service.py` file:

```python
import bentoml

model_runner = bentoml.models.get("text-classification-pipe").to_runner()

svc = bentoml.Service("text-classification-service", runners=[model_runner])

@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def classify(text: str) -> str:
    results = await model_runner.async_run([text])
    return results[0]
```

Now, run the API service locally:

```bash
bentoml serve service.py:svc
```

Sent a prediction request:

```bash
$ curl -X POST -H "Content-Type: text/plain" --data "BentoML is awesome" http://localhost:3000/classify

{"label":"POSITIVE","score":0.9129443168640137}%
```

Define how a [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html) can
be built for deployment, with `bentofile.yaml`:

```yaml
service: 'service.py:svc'
name: text-classification-svc
include:
  - 'service.py'
python:
  packages:
  - torch>=2.0
  - transformers
```

Build a Bento and generate a docker image:

```bash
$ bentoml build
...
Successfully built Bento(tag="text-classification-svc:mc322vaubkuapuqj").
```

```bash
$ bentoml containerize text-classification-svc
Building OCI-compliant image for text-classification-svc:mc322vaubkuapuqj with docker
...
Successfully built Bento container for "text-classification-svc" with tag(s) "text-classification-svc:mc322vaubkuapuqj"
```

```bash
$ docker run -p 3000:3000 text-classification-svc:mc322vaubkuapuqj
```

For a more detailed user guide, check out the
[BentoML Tutorial](https://docs.bentoml.com/en/latest/tutorial.html).

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
