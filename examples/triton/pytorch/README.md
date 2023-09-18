# Triton Inference Server integration

BentoML now provides support for Triton Inference Server.

### Quick tour

Triton Runner can be created via `bentoml.triton.Runner`:

```python
triton_runner = bentoml.triton.Runner(
    "local", model_respository="/path/to/model_repository"
)

svc = bentoml.Service("my-service", runners=[triton_runner])
```

`model_repository` can also accept S3 path:

```python
triton_runner = bentoml.triton.Runner(
    "triton-runners", model_repository="s3://path/to/model_repository"
)
```

CLI arguments can be passed through the `cli_args` argument of `bentoml.triton.Runner`:

```python
triton_runner = bentoml.triton.Runner(
    "triton-runners",
    model_repository="s3://path/to/model_repository",
    cli_args=["--load-model=torchscrip_yolov5s", "--model-control-mode=explicit"],
)
```

An example of inference API:

```python
@svc.api(
    input=bentoml.io.NumpyNdarray.from_sample(
        np.array([[1, 2, 3, 4]]), enforce_dtype=False
    ),
    output=bentoml.io.NumpyNdarray.from_sample(np.array([1])),
)
def triton_infer(input_data: NDArray[np.float16]) -> NDArray[np.int16]:
    iris_res = iris_clf_runner.run(input_data)
    res_kwargs = triton_runner.txt2img.run(IRIS_NAME=iris_res)
    return iris_res
```

Note that each attribute of the `triton_runner` includes the name of all given
models under `model_repository`

APIs from tritonclient are also provided through the Triton Runner:

```python
tritonclient.grpc.aio.InferenceServerClient.get_model_metadata -> triton_runner.get_model_metadata | triton_runner.grpc_get_model_metadata
```

To get build your Bento with Triton, add the following to your `bentofile.yaml`:

```yaml
service: "service.py:svc"
include:
  - *.py
  - /model_repository
docker:
  base_image: nvcr.io/nvidia/tritonserver:22.12-py3
```

To find out more about BentoML Runner architecture, see
[our latest documentation](https://docs.bentoml.org/en/latest/concepts/runner.html#)

For more information about Triton Inference Server, see
[here](https://github.com/triton-inference-server/server)

### Instruction

The following project includes YOLOv5 `TritonRunner` and `bentoml.Runner`.

1. Setup Triton model repository and BentoML model:

```bash
./export-yolov5-weights

python3 train.py
```

2. To build the Bento, use [`build_bento.py`](./build_bento.py):

```bash
python3 build_bento.py
````

> NOTE: To build with custom GPU, pass in `--gpu`. To build with custom tags
> pass in `--tag <custom_tag>`

3. To containerize use [`containerize_bento.py`](./containerize_bento.py):

```bash
python3 containerize_bento.py
```

4. To run the container with Triton, use `docker run`:

```bash
docker run --rm -it -p 3000:3000 triton-integration-pytorch serve-http
```

#### Develop locally:

1. To run Triton locally, do the following:

```bash
BENTOML_GIT_ROOT=$(git rev-parse --show-toplevel)
docker run --rm -it -p 3000-4000:3000-4000 \
           -v $PWD:/workspace -v $BENTOML_GIT_ROOT:/opt/bentoml \
           -v $BENTOML_HOME:/home/bentoml --env BENTOML_HOME=/home/bentoml \
           nvcr.io/nvidia/tritonserver:22.12-py3 bash
```

If you have NVIDIA GPU available, make sure to install
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on your system.
Afterward, passing in `--gpus all` to `docker`:

```bash
BENTOML_GIT_ROOT=$(git rev-parse --show-toplevel)
docker run --gpus all --rm -it -p 3000-4000:3000-4000 \
           -v $PWD:/workspace -v $BENTOML_GIT_ROOT:/opt/bentoml \
           -v $BENTOML_HOME:/home/bentoml --env BENTOML_HOME=/home/bentoml \
           nvcr.io/nvidia/tritonserver:22.12-py3 bash
```

Inside the container shell, there are two options to install BentoML:

- Install from editable

```bash
cd /opt/bentoml && pip install -r requirements/dev-requirements.txt
```

Run the [`setup` script](./setup):

```bash
cd /workspace/ && pip install -r requirements/requirements.txt

bash ./setup
```

2. To serve the Bento, use either `bentoml serve` or
   [`serve_bento.py`](./serve_bento.py) (this requires to have `tritonserver` binary available locally on your system. To use the container, go to step 5)

```bash
python3 serve_bento.py

# bentoml serve-http | serve-grpc triton-integration-pytorch
```

> NOTE: to serve previously custom tag bento, you can also pass in `--tag` to
> `serve_bento.py`


> Feel free to build your own tritonserver. See
> [here](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md)
> for more details on building customisation.

NOTE: If you running into this issue:

````prolog
I0217 00:33:40.955605 3626 server.cc:633]
+---------------------+---------+----------------------------------------------------------------------------------------------------------------------------------------+
| Model               | Version | Status                                                                                                                                 |
+---------------------+---------+----------------------------------------------------------------------------------------------------------------------------------------+
| torchscript_yolov5s | 1       | UNAVAILABLE: Not found: unable to load shared library: /lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block |
+---------------------+---------+----------------------------------------------------------------------------------------------------------------------------------------+

```

Exit out of the process. Then do the following:

```bash
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
```

Then run the `bentoml serve` command again.


<!--
docker run --rm -it -p 3000-3030:3000-3030 -v $(pwd)/model_repository:/models -v ${PWD}:/workspace -v ${BENTOML_GIT_ROOT}:/opt/bentoml -e BENTOML_HOME=/opt/bentoml -v $BENTOML_HOME:/opt/bentoml nvcr.io/nvidia/tritonserver:22.12-py3 bash

cd /opt/bentoml && pip install -r requirements/dev-requirements.txt && cd /workspace && pip install -r requirements/requirements.txt && python3 train.py && ./setup && bentoml serve-http
-->
