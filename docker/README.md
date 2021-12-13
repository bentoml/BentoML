![bentoml-docker](tools/bentoml-docker.png)

## Table of Content
- [Overview](#overview)
- [Basic Usage](#usage)
- [Development](#development)

# Overview

There are three type of BentoServer base image:

| Image Type | Description                                | Supported OS                                          | Usage                             |
|------------|--------------------------------------------|-------------------------------------------------------|-----------------------------------|
| `runtime`  | contains latest BentoML releases from PyPI | `debian`, `centos{7,8}`, `amazonlinux2`, `alpine3.14` | production ready                  |
| `cudnn`    | runtime + support for CUDA-enabled GPU     | `debian`, `centos{7,8}`                               | production ready with GPU support |
| `devel`    | nightly build from development branch      | `debian`, `centos{7,8}`                               | for development use only          |

* Note: currently there's no nightly devel image with GPU support.

The final docker image tags will have the following format:

```markdown
<release_type>-<python_version>-<distros>-<suffix>
   │             │                │        │
   │             │                │        └─> additional suffix, differentiate runtime and cudnn releases
   │             │                └─> formatted <dist><dist_version>, e.g: ami2, debian, centos7
   │             └─> Supported Python version: python3.7 | python3.8 | python3.9
   └─>  Release type: devel or official BentoML release (e.g: 1.0.0)                                           
```

Example image tags:
- `bento-server:devel-python3.7-debian`
- `bento-server:1.0.0-python3.8-centos8-cudnn`
- `bento-server:1.0.0-python3.7-ami2-runtime`

See all available tags [here](https://hub.docker.com/repository/docker/bentoml/bento-server/tags).

# Usage

Before starting:

* Don't edit ephemeral directory: [`generated`](./generated) and [`docs`](./docs)
* Dockerfiles in `./generated` directory must have their build context set to **the directory of this README.md** directory to  add `entrypoint.sh` as well as other helpers files. 
* Every Dockerfile is managed via `manifest.yml` and maintained via `manager.py`, which will render the Dockerfile from `Jinja` templates under `./templates`.

Follow the instructions below to re-generate dockerfiles and build new base images:

```shell

# Build the helper docker images. Refers to Makefile for more information.
» DOCKER_BUILDKIT=1 docker build -t bentoml-docker -f Dockerfile .

# Run the built container with correct users permission for the generated file.
docker run --user $(id -u):$(id -g) -it -v $(pwd):/bentoml bentoml-docker bash 

# Use the provided alias below depending on each tasks.
#
# If you are re-generate Dockerfile you might want to use manager_dockerfiles 
# so that the generated file can have correct permission.
#
# If you are building and pushing Docker images you might want to use manager_images 
# AS ROOT in order to connect to your docker socket mounted to the container
#
# NOTE: Sometimes you might also want to run the following to remove stopped container:
# `docker rm $(docker ps -a -f status=exited -f status=created -q)`
#
# To run verbosely you can choose logs level via -v <loglevel> (eg: -v 5)

alias manager_dockerfiles="docker run --rm -u $(id -u):$(id -g) -v $(pwd):/bentoml bentoml-docker python3 manager.py "

alias manager_images="docker run --rm -v $(pwd):/bentoml -v /var/run/docker.sock:/var/run/docker.sock bentoml-docker python3 manager.py "

# Check manager flags
manager_dockerfiles --helpfull

# To validate generation schema.
manager_dockerfiles --bentoml_version 1.0.0 --validate

# Generate all dockerfiles from templates, and dump all build metadata to metadata.json
manager_dockerfiles --bentoml_version 1.0.0 --generate dockerfiles --dump_metadata --overwrite

# Build all images
manager_images --bentoml_version 1.0.0 --generate images

# Build images for specific releases
manager_images --bentoml_version 1.0.0 --generate images --releases runtime

# Push all images to defined registries under manifest.yml.
manager_images --bentoml_version 1.0.0 --push images --releases cudnn

# Or bring generation and pushing together
manager_images --bentoml_version 1.0.0 --generate images --push images --releases cudnn
```

### Run Locally Built Images

To build each distros releases locally you also need to build a `base` images. This contains all dependencies required by
BentoML before building specific distros images:

```shell
export PYTHON_VERSION=3.8

# with tags for base images, replace the python version to your corresponding python version.
docker build -f ./generated/bento-server/amazonlinux2/Dockerfile \
          --build-arg PYTHON_VERSION=${PYTHON_VERSION} -t bento-server:base-python3.8-ami2 .
```

An example to generate BentoML's AMI base image with `python3.8` that can be used to install `BentoService` and run on AWS Sagemaker:

```shell
# DOCKER_BUILDKIT=1 is optional
DOCKER_BUILDKIT=1 docker build -f ./generated/bento-server/amazonlinux2/runtime/Dockerfile \
                          --build-arg PYTHON_VERSION=${PYTHON_VERSION} -t bentoml-ami2 . 
```

After building the image with tag `bentoml-ami2` (for example), use `docker run` to run the images.

FYI: `-v` (Volume mount) and `-u` (User permission) shares directories and files permission between your local machine and Docker container.
Without `-v` your work will be wiped once container exists, where `-u` will have wrong file permission on your host machine.

```shell
# -v and -u are recommended to use.

# CPU-based images
docker run -i -t -u $(id -u):$(id -g) -v $(pwd)/my-custom-devel bentoml-ami2

# GPU-based images
# See https://docs.bentoml.org/en/latest/guides/gpu_serving.html#general-workaround-recommended
docker run --gpus all --device /dev/nvidia0 --device /dev/nvidiactl \
             --device /dev/nvidia-modeset --device /dev/nvidia-uvm \
             --device /dev/nvidia-uvm-tools -i -t -u $(id -u):$(id -g) -v $(pwd)/my-custom-devel bentoml-ami2
```


# Development

This section covers how BentoML internally manages its docker base image releases.


## Image Manifest

Dockerfile metadata and related configs for multiple platforms are defined under [manifest.yml](../manifest.yml).
This can also be used to control releases with CI pipelines.

The manifest file will be validated while invoking `manager.py`. Refers to [Validation](#validation) for implementations. with the following structure:

- [specs](#specs): holds general templates for our template context
  - [repository](#repository): enable supports for multiple registries: Heroku, Docker Hub, GCR, NVCR.
  - [dependencies](#dependencies): provides options for multiple CUDA releases and others.
  - [releases](#releases): determines releases template for every supported OS.
  - [tag](#tag): determines release tags format
- [packages](#packages): determines supported OS releases for each BentoML's package.
- [distros](#distros): customize values for each releases distros.

#### Image specs
`manifest.yml` structure is defined under `specs`, and others sections on the file can reference to each of sub keys.
See more [here](../manifest.yml).

#### Docker Repository

To determine new registry:
```yaml
repository: &repository_spec
  user: HUB_USERNAME
  pwd: HUB_PASSWORD
  urls:
    api: https://hub.docker.com/v2/users/login/
    repos: https://hub.docker.com/v2/repositories
  registry:
    model-server: docker.io/bentoml/bento-server
```

A registry definition allows us to set up correct docker information to push our final images, tests, and scan.

- `repository_name`: docker.io, gcr.io, ecr.io, etc.

| Keys | Type | defintions |
|------|------|------------|
|`user`| `<str>`| envars for registry username| 
|`pwd`| `<bool>`|envars for registry password| 
|`urls`| `<dict>`| handles the registry API urls|
|`registry`| `<dict>`|handles package registry URIs, which will be parsed by `docker-py`| 

**NOTES**: `urls.api` and `urls.repos` are optional and validated with `cerberus`. We also uses a bit of a hack way 
to update README since we don't setup automatic Docker releases from BentoML's root directory.

#### dependencies

To determine new CUDA version:
```yaml
cuda: &cuda_spec
  cudart:
  cudnn8:
  libcublas:
  libcurand:
  libcusparse:
  libcufft:
  libcusolver:
```

This allows us to add multiple versions of CUDA to support for current DL frameworks as well as backward compatibility

CUDA semver will also be validated. Usually devs should use this key-value pairs as this 
are required library for CUDA and cuDNN. Refers to [cuda's repos](https://developer.download.nvidia.com/compute/cuda/repos/) 
for supported distros and library version.

#### releases

Each of our distros releases will contain the following configs:

| Keys | Type | defintions |
|------|------|------------|
|`templates_dir`| `<str>`|input templates for our distros, can be found [templates/*](../templates)| 
|`base_image`| `<str>`| base distro image: centos:7, debian:buster-slim, etc.| 
|`add_to_tags`| `<str>`|tags suffix to recognize given distro: slim (debian), alpine, ami (amazonlinux)| 
|`multistage_image`| `<bool>`|Enable multistage build (DEFAULT: True)| 
|`header`| `<str>`|headers to be included in our Dockerfile| 
|`envars`| `<list[str]>`|List of environment variables that can be used to build the container images. This will also be checked by our Validator| 
|`cuda_prefix_url`| `<str>`|prefix URL that will be parsed to complete NVIDIA repos for our CUDA dependencies| 
|`cuda_requires`| `<str>`|ported directly from [`nvidia/cuda`](https://hub.docker.com/r/nvidia/cuda) for CUDA devices requirements| 
|`cuda`| `<dict[str,str]>`|list of CUDA components, refers to [CUDA components](#dependencies)| 

#### tag

Docker tag validation can be defined with the following configs:

| Keys | Type | defintions |
|------|------|------------|
|`fmt`| `<str>`|specific python3 string format| 
|`release_type`| `<str>`|devel - runtime - cudnn| 
|`python_version`| `<str>`|3.7 -3.8 - 3.9| 
|`suffixes`| `<str>`|addtional suffix needed to add| 

**NOTES**: The reason why we named our GPU releases `cudnn` instead of `gpu` is due to clarification. With most deep learning framework
they will have some implementationn of NVIDIA's cuDNN libraries. Thus, with `gpu` naming, end users won't have any indication
of having CUDNN in the images, whereas calling `cudnn` will inform users with enough information about the container itself. We also
provide some others math related library from NVIDIA, which are requirements for most deep learning frameworks. You can always add
additional library like [NCCL](https://developer.nvidia.com/nccl), [TensorRT](https://developer.nvidia.com/tensorrt) via the OS package 
manager with your custom docker image built on top of the one BentoML provided.

Tag will also be validated whether keys are defined within the format scope.


## Workflow

High-level workflow when publishing new docker images:

1) validate yaml file
2) generate build context
3) render `j2` templates
4) build from generated directory
5) push to given registries.

Each process will be managed by [``ManagerClient``](../manager.py) with provided functions via `Mixin` class.

The yaml validation process includes:

- Ensure defined spec are valid for the manager scope
- Ensure correct tags formatting with keys-values pair
- Ensure no excess keys are being parsed or setup apart from specs (If you want to include new key-value remember to update `specs` before proceeding.)
- Correct data parsing: envars, https, etc.
