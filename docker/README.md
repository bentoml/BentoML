![bentoml-docker](./bentoml-docker.png)
---
<h2 align="center">BentoML Docker Releases</h2>

[comment]: <comparision between base linux images> (http://crunchtools.com/comparison-linux-container-images/)

### Table of content
- [Notes](#notes)
- [Description](#description)
- [Developing](#developing)

### Notes
- Dockerfiles in `./generated` directory must have their build context set to **the directory of this README.md** directory to  add `entrypoint.sh` as well as other helpers files. 
- Every Dockerfile is managed via `manifest.yml` and maintained via `manager.py`, which will render the Dockerfile from `Jinja` templates under `./templates`.

An example to generate BentoML's AMI base image with `python3.8` that can be used to install `BentoService` and run on AWS Sagemaker:

```shell
» export PYTHON_VERSION=3.8
  
  # DOCKER_BUILDKIT=1 is optional
» DOCKER_BUILDKIT=1 docker build -f ./generated/model-server/amazonlinux2/runtime/Dockerfile \
  --build-args PYTHON_VERSION=${PYTHON_VERSION} -t bentoml-docker . 
```

### Description

For each linux distributions, there will be three type of releases:

| Release Type | Functionality |
|--------------|---------------|
| `runtime`    | contains BentoML latest releases from PyPI |
| `cudnn`      | runtime + CUDA and CUDNN  for GPU support |
| `devel`      | nightly build directly from `master` branch |

### Developing

To add new distros support or new CUDA version, you first have to update `manifest.yml`, add templates with correct format under `./templates`, then run `manager.py` to re-generate new Dockerfiles.

You can use the provided [Dockerfile](https://github.com/bentoml/BentoML/blob/master/docker/Dockerfile) to have a fully installed environment.
```shell

# Build the helper docker images. Refers to Makefile for more information.
» make install

# Run the built container with correct users permission for the generated file.
» make run 

# Use the provided alias below depending on each tasks.
#
# If you are re-generate Dockerfile you might want to use manager_dockerfiles so that the generated file can have correct permission
#
# If you are building and pushing Docker images you might want to use manager_images AS ROOT in order to connect to your docker socket mounted to the container
#
# This is for rebuilding or adding new Dockerfile
» alias manager_dockerfiles="docker run --rm -u $(id -u):$(id -g) -v $(pwd):/bentoml bentoml-docker python3 manager.py "

# When building or deploying images you need to run as root
» alias manager_images="docker run --rm -v $(pwd):/bentoml -v /var/run/docker.sock:/var/run/docker.sock bentoml-docker python3 manager.py "

# Generate all dockerfiles from templates, and dump all build metadata to metadata.json
» manager_dockerfiles -bv 0.13.0 -dm

# Build all images
» manager_images -bv 0.13.0 -bi 

# Push all images to defined registries under manifest.yml
» manager_images -bv 0.13.0 -pth
```

#### Environment

#### Manifest