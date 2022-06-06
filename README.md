[<img src="https://raw.githubusercontent.com/bentoml/BentoML/main/docs/source/_static/img/bentoml-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/bentoml/BentoML)
<br>

# The Unified Model Serving Framework  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20The%20Unified%20Model%20Serving%20Framework%20&url=https://github.com/bentoml&via=bentomlai&hashtags=mlops,bentoml)

[![pypi_status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![actions_status](https://github.com/bentoml/bentoml/workflows/CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.bentoml.org)

BentoML simplifies ML model deployment and serves your models at production scale.

👉 [Join us in our Slack community](https://l.linklyhq.com/l/ktOh) where hundreds of ML 
practitioners are contributing to the project, helping other users, and discuss all 
things MLOps.

## Feature Highlights ##

✨ Model Serving the way you need it 
- **Online serving** via REST API or gRPC
- **Offline scoring** on batch datasets with Apache Spark, or Dask.
- **Stream serving** with Kafka, Beam, and Flink

🍱 Easily go from training to model serving in production
- All common ML libraries natively supported! - Tensorflow, PyTorch, XGBoost, Scikit-Learn and many more
- Standard `.bento` format for packaging code, models and dependencies for easy versioning and deployment
- Automatically setup Dockerfile, CUDA and python packages for model serving 
- Designed to work with any training pipeline or experimentation platform

🐍 Python-first, scales with powerful optimizations
- Scale model inference workers separately from business logic and feature processing code
- Adaptive batching dynamically groups inference requests for optimal performance
- Inference graphs with multiple models automatically orchestrated with Yatai on Kubernetes

🚢 Deployment workflow made for production
- 🐳 Automatically generate docker images for production deployment
- [🦄️ Yatai](https://github.com/bentoml/yatai): Model Deployment at scale on Kubernetes
- [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on any cloud platform

## Getting Started ##

__The BentoML version 1.0 is in pre-view release.__ You can be of great help by testing out the preview release, 
reporting issues, contribute to the documentation and create sample gallery projects.  For our most recent stable release, see the [0.13-LTS branch](https://github.com/bentoml/BentoML/tree/0.13-LTS).

- [Documentation](https://docs.bentoml.org/) - Overview of the BentoML docs and related resources
- [Tutorial: Intro to BentoML](https://docs.bentoml.org/en/latest/tutorial.html) - Learn by doing! In under 10 minutes, you'll serve a model via REST API and generate a docker image for deployment.
- [Main Concepts](https://docs.bentoml.org/en/latest/concepts/index.html) - A step-by-step tour for learning main concepts in BentoML
- [Examples](https://github.com/bentoml/gallery) - Gallery of sample projects using BentoML
- [ML Framework Sepecific Guides](https://docs.bentoml.org/en/latest/frameworks/index.html) - Best practices and example usages by the ML framework of your choice
- [Advanced Guides](https://docs.bentoml.org/en/latest/guides/index.html) - Learn about BentoML's internals, architecture and advanced features


## Installation ##

```bash
pip install bentoml --pre
```

## Quick Tour ##

**Step 1:** At the end of your model training pipeline, save your trained model instance with BentoML:

```python
import bentoml

model = train(...)

bentoml.pytorch.save_model("fraud_detect", model)

# INFO  [cli] Using default model signature `{"predict": {"batchable": False}}` for pytorch model
# INFO  [cli] Successfully saved Model(tag="fraud_detect:3qee3zd7lc4avuqj", path="~/bentoml/models/fraud_detect/3qee3zd7lc4avuqj/")
```

BentoML saves the model artifact files in a local model store, a long with necessary metadata. 
A new version tag is automatically generated for the model.

Optionally, you may provide the `signatures` of your model for running inference with
dynamic batching enabled, and attach labels, metadata, or `custom_objects` to be saved 
together with your model, e.g.:

```python
bentoml.pytorch.save_model(
    "demo_mnist",  # model name in the local model store
    trained_model,  # model instance being saved
    signatures={   # model signatures for runner inference
      "predict": {
        "batchable": True,
        "batch_dim": 0,
      }
    },
    metadata={   # user-defined additional metadata
        "acc": acc,
        "cv_stats": cv_stats,
    },
)
```


**Step 2:** Create a prediction service with the saved model:

Create a `service.py` file with:
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

Start an HTTP server locally:

```bash
bentoml serve service.py:svc
```

And sent a test request to it:
```bash
curl -F 'image=@samples/1.png' http://127.0.0.1:3000/predict_image
```

You can also open http://127.0.0.1:3000 in a browser and debug the endpoint by sending
requests directly from the web UI.

Note that saved model is converted into a `Runner`, which in BentoML, represents a unit 
of computation that can be scaled separately. In local deployment mode, this means the 
model will be running in its own worker processes. Since the model is saved with a 
`batchable: True` signature, BentoML applies dynamic batching to all the 
`mnist_runner.predict.run` calls under the hood for optimal performance.


**Step 3:** Build a Bento for deployment:

Define a `bentofile.yaml` build file for your ML project:

```yaml
service: "service:svc"  # where the bentoml.Service instance is defined
include:
- "*.py"
exclude:
- "tests/"
python:
  packages:
    - numpy
    - torch
    - Pillow
docker:
  distro: debian
  cuda_version: 11.6.2
```

Build a `Bento` using the `bentofile.yaml` specification from current directory: 
```bash
> bentoml build

INFO [cli] Building BentoML service "pytorch_mnist:4mymorgurocxjuqj" from build context "~/workspace/gallery/pytorch_mnist"
INFO [cli] Packing model "demo_mnist:7drxqvwsu6zq5uqj" from "~/bentoml/models/demo_mnist/7drxqvwsu6zq5uqj"
INFO [cli] Locking PyPI package versions..
INFO [cli]
           ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
           ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
           ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
           ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
           ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
           ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

INFO [cli] Successfully built Bento(tag="pytorch_mnist:4mymorgurocxjuqj") at "~/bentoml/bentos/pytorch_mnist/4mymorgurocxjuqj/"
```
The Bento with `tag="pytorch_mnist:4mymorgurocxjuqj"` is now created in the local `Bento` store. It is an archive containing all the source code, model files, and dependency specs - anything that is required for reproducing the model in an identical environment for serving in production.


**Step 4:** Deploying the `Bento`

Generate a docker image from the Bento and run a docker container locally for serving:
```bash
> bentoml containerize pytorch_mnist:4mymorgurocxjuqj

INFO [cli] Successfully built docker image "pytorch_mnist:4mymorgurocxjuqj"

> docker run --gpus all -p 3000:3000 pytorch_mnist:4mymorgurocxjuqj
```

Learn more about other deployment options [here](https://docs.bentoml.org/en/latest/concepts/deploy.html).


## Community ##

- For general questions and support, join the [community slack](https://l.linklyhq.com/l/ktOh).
- To receive release notification, star & watch the BentoML project on GitHub.
- To report a bug or suggest a feature request, use [GitHub Issues](https://github.com/bentoml/BentoML/issues/new/choose).
- For long-form discussions, use [Github Discussions](https://github.com/bentoml/BentoML/discussions).
- To stay informed with community updates, follow the [BentoML Blog](modelserving.com) and [@bentomlai](http://twitter.com/bentomlai) on Twitter.


## Contributing ##

There are many ways to contribute to the project:

- If you have any feedback on the project, share it in [Github Discussions](https://github.com/bentoml/BentoML/discussions) or the `#bentoml-contributors` channel in the [community slack](https://l.linklyhq.com/l/ktOh).
- Report issues you're facing and "Thumbs up" on issues and feature requests that are relevant to you.
- Investigate bugs and reviewing other developer's pull requests.
- Contributing code or documentation to the project by submitting a Github pull request. Check out the [Development Guide](https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md).
- Learn more in the [contributing guide](https://github.com/bentoml/BentoML/blob/main/CONTRIBUTING.md).

### Contributors! ###

Thanks to all of our amazing contributors!

<a href="https://github.com/bentoml/BentoML/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bentoml/BentoML" />
</a>

---

### Usage Reporting ###

BentoML collects anonymous usage data that helps our team to improve the product.
Only BentoML's internal API calls are being reported. We strip out as much potentially 
sensitive information as possible, and we will never collect user code, model data, model names, or stack traces.
Here's the [code](./bentoml/_internal/utils/analytics/usage_stats.py) for usage tracking.
You can opt-out of usage tracking by the `--do-not-track` CLI option:
```bash
bentoml [command] --do-not-track
```

Or by setting environment variable `BENTOML_DO_NOT_TRACK=True`:
```bash
export BENTOML_DO_NOT_TRACK=True
```
---

### License ###

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/main/LICENSE)

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=small)](https://app.fossa.com/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_small)
