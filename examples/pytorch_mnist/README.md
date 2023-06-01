# BentoML PyTorch MNIST Tutorial

This is a sample project demonstrating basic usage of BentoML, the machine learning model serving library.

In this project, we will train a digit recognition model using PyTorch on the MNIST dataset, build
an ML service for the model, serve the model behind an HTTP endpoint, and containerize the model
server as a docker image for production deployment.

This project is also available to run from a notebook: https://github.com/bentoml/BentoML/blob/main/examples/pytorch_mnist/pytorch_mnist_demo.ipynb

### Install Dependencies

Install python packages required for running this project:

```bash
pip install -r ./requirements.txt
```

### Model Training

First step, train a digit recognition model with PyTorch using BentoML:

```bash
python train.py
```

This should save a new model in the BentoML local model store:

```bash
bentoml models list
```

Verify that the model can be loaded as runner from Python shell:

```python
import numpy as np
import PIL.Image
import torch

import bentoml

runner = bentoml.pytorch.get("pytorch_mnist:latest").to_runner()
runner.init_local()

img = PIL.Image.open("samples/0.png")
np_img = np.array(img)
tensor_img = torch.from_numpy(np_img).float()
tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
tensor_img = torch.nn.functional.interpolate(tensor_img, size=28, mode='bicubic', align_corners=False)

result = runner.predict.run(tensor_img)  # => tensor(0)
```

### Create ML Service

The ML Service code is defined in the `service.py` file:

```python
# service.py
import typing as t

import numpy as np
import PIL.Image
from PIL.Image import Image as PILImage

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray


mnist_runner = bentoml.pytorch.get(
    "pytorch_mnist",
    name="mnist_runner",
    predict_fn_name="predict",
).to_runner()

svc = bentoml.Service(
    name="pytorch_mnist_demo",
    runners=[
        mnist_runner,
    ],
)


@svc.api(
    input=NumpyNdarray(dtype="float32", enforce_dtype=True),
    output=NumpyNdarray(dtype="int64"),
)
async def predict_ndarray(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.shape == (28, 28)
    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension
    inp = np.expand_dims(inp, 0)
    output_tensor = await mnist_runner.async_run(inp)
    return output_tensor.numpy()


@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    arr = np.array(f)/255.0
    assert arr.shape == (28, 28)

    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension
    arr = np.expand_dims(arr, 0).astype("float32")
    output_tensor = await mnist_runner.async_run(arr)
    return output_tensor.numpy()
```

We defined two api endpoints `/predict_ndarray` and `/predict_image` with single runner.

Start an API server locally to test the service code above:

```bash
bentoml serve service:svc --reload
```

With the `--reload` flag, the API server will automatically restart when the source
file `service.py` is being edited, to boost your development productivity.

Verify the endpoint can be accessed locally:

```bash
curl -H "Content-Type: multipart/form-data" -F'fileobj=@samples/1.png;type=image/png' http://127.0.0.1:3000/predict_image
```

We can also do a simple local benchmark if [locust](https://locust.io) is installed:

```bash
locust --headless -u 100 -r 1000 --run-time 10m --host http://127.0.0.1:3000
```

### Build Bento for deployment

A `bentofile` is already created in this directory for building a
Bento for the service:

```yaml
service: "service:svc"
description: "file: ./README.md"
labels:
  owner: bentoml-team
  stage: demo
include:
  - "*.py"
exclude:
  - "tests/"
python:
  packages:
    - scikit-learn
    - torch
    - Pillow
```

Note that we exclude `tests/` from the bento using `exclude`.

Simply run `bentoml build` from current directory to build a Bento with the latest
version of the `pytorch_mnist` model. This may take a while when running for the first
time for BentoML to resolve all dependency versions:

```
> bentoml build

[01:14:04 AM] INFO     Building BentoML service "pytorch_mnist_demo:bmygukdtzpy6zlc5vcqvsoywq" from build context
                       "/home/chef/workspace/gallery/pytorch"
              INFO     Packing model "pytorch_mnist_demo:xm6jsddtu3y6zluuvcqvsoywq" from
                       "/home/chef/bentoml/models/pytorch_mnist_demo/xm6jsddtu3y6zluuvcqvsoywq"
              INFO     Locking PyPI package versions..
[01:14:05 AM] INFO
                       ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
                       ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
                       ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
                       ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
                       ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
                       ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

              INFO     Successfully built Bento(tag="pytorch_mnist_demo:bmygukdtzpy6zlc5vcqvsoywq") at
                       "/home/chef/bentoml/bentos/pytorch_mnist_demo/bmygukdtzpy6zlc5vcqvsoywq/"
```

This Bento can now be loaded for serving:

```bash
bentoml serve pytorch_mnist_demo:latest --production
```

The Bento directory contains all code, files, models and configs required for running this service.
BentoML standarlizes this file structure which enables serving runtimes and deployment tools to be
built on top of it. By default, Bentos are managed under the `~/bentoml/bentos` directory:

```
> cd ~/bentoml/bentos/pytorch_mnist_demo && cd $(cat latest)

> tree
.
├── apis
│   └── openapi.yaml
├── bento.yaml
├── env
│   ├── conda
│   ├── docker
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   └── init.sh
│   └── python
│       ├── requirements.lock.txt
│       ├── requirements.txt
│       └── version.txt
├── models
│   └── pytorch_mnist_demo
│       ├── eqxdigtybch6nkfb
│       │   ├── model.yaml
│       │   └── saved_model.pt
│       └── latest
├── README.md
└── src
    ├── model.py
    ├── service.py
    └── train.py

9 directories, 15 files
```

### Containerize Bento for deployment

Make sure you have docker installed and docker deamon running, and the following command
will use your local docker environment to build a new docker image, containing the model
server configured from this Bento:

```bash
bentoml containerize pytorch_mnist_demo:latest
```

Test out the docker image built:

```bash
docker run -P pytorch_mnist_demo:invwzzsw7li6zckb2ie5eubhd
```
