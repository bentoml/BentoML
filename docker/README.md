![bentoml-docker](./bentoml-docker.png)
---
<h2 align="center">BentoML Docker Releases</h2>

[comment]: <comparision between base linux images> (http://crunchtools.com/comparison-linux-container-images/)

### Notes
- Dockerfiles in `./generated` directory must have their build context set to **the directory of this README.md** directory to  add `entrypoint.sh` as well as other helpers files. 
- Every Dockerfile is managed via `manifest.yml` and maintained via `manager.py`, which will render the Dockerfile from `Jinja` templates under `./templates`.

An example to generate BentoML's AMI base image with `python3.8` that can be used to install `BentoService` and run on AWS Sagemaker:

```shell
    » export PYTHON_VERSION=3.8
      
      # DOCKER_BUILDKIT=1 is optional
    » DOCKER_BUILDKIT=1 docker build -f ./generated/model-server/amazonlinux2/runtime/Dockerfile --build-args PYTHON_VERSION=${PYTHON_VERSION} -t bentoml-docker . 
```

### Description

For each linux distributions, there will be three type of releases:

| Release Type | Functionality |
|--------------|---------------|
| runtime      | contains BentoML latest releases from PyPI |
| cudnn        | runtime + CUDA and CUDNN preinstalled for GPU support |
| devel        | nightly build directly from master branch |

### Developing
```shell
    » alias manager_dockerfiles="docker run --rm -u $(id -u):$(id -g) -v $(pwd):/bentoml bentoml-docker python3 manager.py "
    
    » alias manager_images="docker run --rm -v $(pwd):/bentoml -v /var/run/docker.sock:/var/run/docker.sock bentoml-docker python3 manager.py "

```