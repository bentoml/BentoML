# BentoML TensorFlow 2 Tutorial

This is a sample project demonstrating usage of BentoML following the advanced TensorFlow2 quickstart here: https://www.tensorflow.org/tutorials/quickstart/advanced

In this project, will train a model using Tensorflow2 library and the MNIST dataset. We will then build
an ML service for the model, serve the model behind an HTTP endpoint, and containerize the model
server as a docker image for production deployment.

This project is also available to run from a notebook: https://github.com/bentoml/BentoML/blob/main/examples/tensorflow2/tensorflow2_mnist_demo.ipynb

### Install Dependencies

First install the requirements for this guide
```bash
pip install -r requirements.txt
```

For MacOS 11+
```bash
pip install -r requirements-macos.txt
```

At the time of this writing, for M1 Macbooks, if you are getting the following error:
```bash
ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based projects
```
Then you'll need to install and configure the h5py library through brew like this:
```bash
brew install hdf5
export CPATH="/opt/homebrew/include/"
export HDF5_DIR=/opt/homebrew/
```

Then try running the "pip install tensorflow-macos" again 


### Model Training

First step, train a classification model with tensorflow's built in mnist dataset and save the model
with BentoML:

```bash
python train.py
```

If you look at the last line of train.py, you'll see:
````python
bentoml.tensorflow.save_model("tensorflow_mnist", model)
````


This should save a new model in the BentoML local model store under model name "tensorflow_mnist:

```bash
bentoml models list
```

Verify that the model can be loaded as runner from Python shell:

```python
import numpy as np
import PIL.Image

import bentoml

runner = bentoml.tensorflow.get("tensorflow_mnist:latest").to_runner()
runner.init_local()  # for debug only. please do not call this in the service

img = PIL.Image.open("samples/0.png")
arr = np.array(img) / 255.0
arr = arr.astype("float32")

# add color channel dimension for greyscale image
arr = np.expand_dims(arr, 2)
runner.run(arr)  # => returns an array of probabilities for numbers 0-9
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


mnist_runner = bentoml.tensorflow.get("tensorflow_mnist").to_runner()

svc = bentoml.Service(
    name="tensorflow_mnist_demo",
    runners=[
        mnist_runner,
    ],
)


@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    arr = np.array(f)/255.0
    assert arr.shape == (28, 28)

    # extra channel dimension
    arr = np.expand_dims(arr, (0, 3)).astype("float32")
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
curl -H "Content-Type: multipart/form-data" -F'fileobj=@samples/0.png;type=image/png' http://127.0.0.1:3000/predict_image
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
- "locustfile.py"
python:
  lock_packages: false
  packages:
    - tensorflow
```

Note that we exclude `locustfile.py` from the bento using `exclude`.

Simply run `bentoml build` from current directory to build a Bento with the latest
version of the `tensorflow_mnist` model.

This may take a while when running for the first time for BentoML to resolve all dependency versions:

```
> bentoml build

[01:14:04 AM] INFO     Building BentoML service "tensorflow_mnist_demo:bmygukdtzpy6zlc5vcqvsoywq" from build context      
                       "/home/chef/workspace/gallery/tensorflow2"                                                         
              INFO     Packing model "tensorflow_mnist_demo:xm6jsddtu3y6zluuvcqvsoywq" from                               
                       "/home/chef/bentoml/models/tensorflow_mnist_demo/xm6jsddtu3y6zluuvcqvsoywq"                       
              INFO     Locking PyPI package versions..                                                                 
[01:14:05 AM] INFO                                                                                                     
                       ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░                                   
                       ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░                                   
                       ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░                                   
                       ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░                                   
                       ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗                                   
                       ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝                                   
                                                                                                                       
              INFO     Successfully built Bento(tag="tensorflow_mnist_demo:bmygukdtzpy6zlc5vcqvsoywq") at                 
                       "/home/chef/bentoml/bentos/tensorflow_mnist_demo/bmygukdtzpy6zlc5vcqvsoywq/"                      
```

This Bento can now be loaded for serving:

```bash
bentoml serve tensorflow_mnist_demo:latest --production
```

The Bento directory contains all code, files, models and configs required for running this service.
BentoML standarlizes this file structure which enables serving runtimes and deployment tools to be
built on top of it. By default, Bentos are managed under the `~/bentoml/bentos` directory:

```
> cd ~/bentoml/bentos/tensorflow_mnist_demo && cd $(cat latest)

> tree
.
├── README.md
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
│   └── tensorflow_mnist
│       ├── latest
│       └── wz77wdeuegyh2du5
│           ├── assets
│           ├── model.yaml
│           ├── saved_model.pb
│           └── variables
│               ├── variables.data-00000-of-00001
│               └── variables.index
└── src
    ├── service.py
    └── train.py

11 directories, 16 files
```


### Containerize Bento for deployment

Make sure you have docker installed and docker deamon running, and the following command
will use your local docker environment to build a new docker image, containing the model
server configured from this Bento:

```bash
bentoml containerize tensorflow_mnist_demo:latest
```

Test out the docker image built:
```bash
docker run -p 3000:3000 tensorflow_mnist_demo:latest
```
