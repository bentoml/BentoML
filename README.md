[<img src="https://raw.githubusercontent.com/bentoml/BentoML/main/docs/source/_static/img/bentoml-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/bentoml/BentoML)
<br>

# The Unified Model Serving Framework [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20The%20Unified%20Model%20Serving%20Framework%20&url=https://github.com/bentoml&via=bentomlai&hashtags=mlops,bentoml)

[![pypi_status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![actions_status](https://github.com/bentoml/bentoml/workflows/CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.bentoml.org)

BentoML makes it easy to create Machine Learning services that are ready to deploy and scale.

👉 [Join our Slack community today!](https://l.bentoml.com/join-slack)

✨ Looking deploy your ML service quickly? Checkout [BentoML Cloud](https://l.bentoml.com/bento-cloud)
for the easiest and fastest way to deploy your bento.

## Getting Started

- [Documentation](https://docs.bentoml.org/) - Overview of the BentoML docs and related resources
- [Tutorial: Intro to BentoML](https://docs.bentoml.org/en/latest/tutorial.html) - Learn by doing! In under 10 minutes, you'll serve a model via REST API and generate a docker image for deployment.
- [Main Concepts](https://docs.bentoml.org/en/latest/concepts/index.html) - A step-by-step tour for learning main concepts in BentoML
- [Examples](https://github.com/bentoml/BentoML/tree/main/examples) - Gallery of sample projects using BentoML
- [ML Framework Guides](https://docs.bentoml.org/en/latest/frameworks/index.html) - Best practices and example usages by the ML framework of your choice
- [Advanced Guides](https://docs.bentoml.org/en/latest/guides/index.html) - Learn about BentoML's internals, architecture and advanced features
- Need help? [Join BentoML Community Slack 💬](https://l.linklyhq.com/l/ktOh)

---

## Highlights

🍭 Unified Model Serving API

- Framework-agnostic model packaging for Tensorflow, PyTorch, XGBoost, Scikit-Learn, ONNX, and [many more](https://docs.bentoml.org/en/latest/frameworks/index.html)!
- Write **custom Python code** alongside model inference for pre/post-processing and business logic
- Apply the **same code** for online(REST API or gRPC), offline batch, and streaming inference
- Simple abstractions for building **multi-model inference** pipelines or graphs

🚂 **Standardized process** for a frictionless transition to production

- Build [Bento](https://docs.bentoml.org/en/latest/concepts/bento.html) as the standard deployable artifact for ML services
- Automatically **generate docker images** with the desired dependencies
- Easy CUDA setup for inference with GPU
- Rich integration with the MLOps ecosystem, including Kubeflow, Airflow, MLFlow, Triton

🏹 **_Scalable_** with powerful performance optimizations

- [Adaptive batching](https://docs.bentoml.org/en/latest/guides/batching.html) dynamically groups inference requests on server-side optimal performance
- [Runner](https://docs.bentoml.org/en/latest/concepts/runner.html) abstraction scales model inference separately from your custom code
- [Maximize your GPU](https://docs.bentoml.org/en/latest/guides/gpu.html) and multi-core CPU utilization with automatic provisioning

🎯 Deploy anywhere in a **DevOps-friendly** way

- Streamline production deployment workflow via:
  - [☁️ BentoML Cloud](https://l.bentoml.com/bento-cloud): the fastest way to deploy your bento, simple and at scale
  - [🦄️ Yatai](https://github.com/bentoml/yatai): Model Deployment at scale on Kubernetes
  - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on AWS SageMaker, Lambda, ECE, GCP, Azure, Heroku, and more!
- Run offline batch inference jobs with Spark or Dask
- Built-in support for Prometheus metrics and OpenTelemetry
- Flexible APIs for advanced CI/CD workflows

## How it works

Save your trained model with BentoML:

```python
import bentoml

saved_model = bentoml.pytorch.save_model(
    "demo_mnist", # model name in the local model store
    model, # model instance being saved
)

print(f"Model saved: {saved_model}")
# Model saved: Model(tag="demo_mnist:3qee3zd7lc4avuqj", path="~/bentoml/models/demo_mnist/3qee3zd7lc4avuqj/")
```

Define a prediction service in a `service.py` file:

```python
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image
from PIL.Image import Image as PILImage

mnist_runner = bentoml.pytorch.get("demo_mnist:latest").to_runner()

svc = bentoml.Service("pytorch_mnist", runners=[mnist_runner])

@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
def predict(input_img: PILImage):
    img_arr = np.array(input_img)/255.0
    input_arr = np.expand_dims(img_arr, 0).astype("float32")
    output_tensor = mnist_runner.predict.run(input_arr)
    return output_tensor.numpy()
```

Create a `bentofile.yaml` build file for your ML service:

```yaml
service: "service:svc"
include:
  - "*.py"
python:
  packages:
    - numpy
    - torch
    - Pillow
```

Now, run the prediction service:

```bash
bentoml serve
```

Sent a prediction request:

```bash
curl -F 'image=@samples/1.png' http://127.0.0.1:3000/predict_image
```

Build a Bento and generate a docker image:

```bash
$ bentoml build
Successfully built Bento(tag="pytorch_mnist:4mymorgurocxjuqj") at "~/bentoml/bentos/pytorch_mnist/4mymorgurocxjuqj/"

$ bentoml containerize pytorch_mnist:4mymorgurocxjuqj
Successfully built docker image "pytorch_mnist:4mymorgurocxjuqj"

$ docker run -p 3000:3000 pytorch_mnist:4mymorgurocxjuqj
Starting production BentoServer from "pytorch_mnist:4mymorgurocxjuqj" running on http://0.0.0.0:3000
```

For a more detailed user guide, check out the [BentoML Tutorial](https://docs.bentoml.org/en/latest/tutorial.html).

---

## Community

- For general questions and support, join the [community slack](https://l.linklyhq.com/l/ktOh).
- To receive release notification, star & watch the BentoML project on [GitHub](https://github.com/bentoml/BentoML).
- To report a bug or suggest a feature request, use [GitHub Issues](https://github.com/bentoml/BentoML/issues/new/choose).
- To stay informed with community updates, follow the [BentoML Blog](http://modelserving.com) and [@bentomlai](http://twitter.com/bentomlai) on Twitter.

## Contributing

There are many ways to contribute to the project:

- If you have any feedback on the project, share it under the `#bentoml-contributors` channel in the [community slack](https://l.linklyhq.com/l/ktOh).
- Report issues you're facing and "Thumbs up" on issues and feature requests that are relevant to you.
- Investigate bugs and reviewing other developer's pull requests.
- Contributing code or documentation to the project by submitting a GitHub pull request. Check out the [Development Guide](https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md).
- Learn more in the [contributing guide](https://github.com/bentoml/BentoML/blob/main/CONTRIBUTING.md).

### Contributors

Thanks to all of our amazing contributors!

<a href="https://github.com/bentoml/BentoML/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bentoml/BentoML" />
</a>

---

### Usage Reporting

BentoML collects usage data that helps our team to improve the product.
Only BentoML's internal API calls are being reported. We strip out as much potentially
sensitive information as possible, and we will never collect user code, model data, model names, or stack traces.
Here's the [code](./src/bentoml/_internal/utils/analytics/usage_stats.py) for usage tracking.
You can opt-out of usage tracking by the `--do-not-track` CLI option:

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
