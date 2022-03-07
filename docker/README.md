![bentoml-docker](hack/bentoml-docker.png)

BentoML is an open platform that simplifies ML model deployment and enables you to serve your models at production scale in minutes

👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktOr) We're happy to help with any issue you face or even just to meet you and hear what you're working on :)

## Table of Content
- [Overview](#overview)
- [Basic Usage](#usage)
- [Development](#development)

# Overview

There are three type of BentoServer base image:

| Image Type | Description                                | Supported OS                                          | Usage                             |
|------------|--------------------------------------------|-------------------------------------------------------|-----------------------------------|
| `runtime`  | contains latest BentoML releases from PyPI | `debian`, `ubi{7,8}`, `amazonlinux2`, `alpine3.14` | production ready                  |
| `cudnn`    | runtime + support for CUDA-enabled GPU     | `debian`, `ubi{7,8}`                               | production ready with GPU support |
| `devel`    | nightly build from development branch      | `debian`, `ubi{7,8}`                               | for development use only          |

* Note: currently there's no nightly devel image with GPU support.

The final docker image tags will have the following format:

```markdown
<release_type>-<python_version>-<distros>-<suffix>
   │             │                │        │
   │             │                │        └─> additional suffix, differentiate runtime and cudnn releases
   │             │                └─> formatted <dist><dist_version>, e.g: ami2, debian, ubi7
   │             └─> Supported Python version: python3.7 | python3.8 | python3.9
   └─>  Release type: devel or official BentoML release (e.g: 1.0.0)                                           
```

Example image tags:
- `bento-server:devel-python3.7-debian`
- `bento-server:1.0.0-python3.8-ubi8-cudnn`
- `bento-server:1.0.0-python3.7-ami2-runtime`

See all available tags [here](https://hub.docker.com/repository/docker/bentoml/bento-server/tags).

# Usage

Before starting:

* We will utilize `docker buildx` underthe hood, so make sure to have that installed with `docker buildx install` (buildx should already installed for folks with Docker > 19.03)

* we will also use [docker-pushrm](https://github.com/christian-korneck/docker-pushrm) to push the given readme, so make sure to have that installed as well.

* In order to setup for multiple architecture [QEMU](https://github.com/multiarch/qemu-user-static) is required. Install with:
```bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

* Don't edit ephemeral directory: [`generated`](./generated) and [`docs`](./docs)
* Dockerfiles in `./generated` directory must have their build context set to **the directory of this README.md** directory to  add `entrypoint.sh` as well as other helpers files. 
* Every Dockerfile is managed via `manifest.yml` and maintained via `manager.py`, which will render the Dockerfile from `Jinja` templates under `./templates`.
* Refers to [Docker's Official Images](https://github.com/docker-library/official-images) for multi-architecture support.

Follow the instructions below to re-generate dockerfiles and build new base images:


Build the helper docker images. Refers to Makefile for more information.
```shell
» DOCKER_BUILDKIT=1 docker build -t bentoml-docker -f Dockerfile .
```

Run the built container with correct users permission for the generated file.
```shell
» docker run --user $(id -u):$(id -g) -it -v $(pwd):/bentoml bentoml-docker bash 
```

NOTE: Sometimes you might also want to run the following to remove stopped container:
`docker rm $(docker ps -a -f status=exited -f status=created -q)`

To run verbosely you can choose logs level via -v <loglevel> (eg: -v 5)

Now manager is a CLI tool :smile:. Install it with:
```bash
» pip install -e .
```
Or run directly from the newly built docker container:
```bash
» docker run --init --rm -u $(id -u):$(id -g) -v $GIT_ROOT/docker:/bentoml bentoml-docker manager $@
```

Check manager flags
```bash
» manager --help
Usage: manager [OPTIONS] COMMAND [ARGS]...

        Manager: BentoML's Docker Images release management system.

        Features:

            📝 Multiple Python version: 3.7, 3.8, 3.9+, ...
            📝 Multiple platform: arm64v8, amd64, ppc64le, ...
            📝 Multiple Linux Distros that you love: Debian, Ubuntu, UBI, alpine, ...

        Get started with:
            $ manager --help


Options:
  -v, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  authenticate  Authenticate with a given registry.
  build         Build and Release Docker images.
  generate      Generate files for a given docker image.
```


Generate all dockerfiles from templates, and dump all build metadata to `build.meta.json` and `releases.meta.json`
```bash
» manager generate --bentoml_version 1.0.0a5 --dump_metadata
```

Authenticate with a given registry:
```bash
manager authenticate --registry docker.io
```

Build all images with supoprts for multiple platform by default:
```bash
manager build --bentoml_version 1.0.0a5 --max_workers 5
```

Build images for specific releases
```bash
manager build --bentoml_version 1.0.0a --releases runtime
```

Since we are utilizing `docker buildx`, `--push` is enabled by default when
running build, so sit back and relax.


### Run Locally Built Images

To build out an image from generated Dockerfile, do:

```shell
PYTHON_VERSION=3.8
BENTOML_VERSION=1.0.0a6
IMAGE_NAME="bentoml/bento-server"
CUDA_VERSION="11.4.1"
OS="ubi8"
ARCHES="x86_64, arm64v8, ppc64le, s390x"
PLATFORM_ARGS=`printf '%s ' '--platform'; for var in $(echo $ARCHES | sed "s/,/ /g"); do printf 'linux/%s,' "$var"; done | sed 's/,*$//g'`

docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:base-python${PYTHON_VERSION}-${OS} \
    -f ./generated/bento-server/${OS}/base/Dockerfile \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} .

docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:${BENTOML_VERSION}-python${PYTHON_VERSION}-${OS}-runtime \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} ./generated/bento-server/${OS}/runtime

docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:${BENTOML_VERSION}-python${PYTHON_VERSION}-${OS}-cudnn \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} ./generated/bento-server/${OS}/cudnn

docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:devel-python${PYTHON_VERSION}-${OS} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} ./generated/bento-server/${OS}/devel
```

FYI: `-v` (Volume mount) and `-u` (User permission) shares directories and files permission between your local machine and Docker container.
Without `-v` your work will be wiped once container exists, where `-u` will have wrong file permission on your host machine.

```shell
# -v and -u are recommended to use.

# CPU-based images
docker run -it -u $(id -u):$(id -g) -v $(pwd):/bentoml bentoml/bento-server:1.0.0a6-python3.8-ubi8-runtime

# GPU-based images
# See https://docs.bentoml.org/en/latest/guides/gpu_serving.html#general-workaround-recommended
docker run --gpus all --device /dev/nvidia0 --device /dev/nvidiactl \
             --device /dev/nvidia-modeset --device /dev/nvidia-uvm \
             --device /dev/nvidia-uvm-tools -i -t -u $(id -u):$(id -g) \
             -v $(pwd):/bentoml bentoml/bento-server:1.0.0a6-python3.8-ubi8-runtime
```


# Development

This section covers how BentoML internally manages its docker base image releases.
NOTE: this part is currently actively WIP, since I just refactor the system.


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
|`base_image`| `<str>`| base distro image: ubi:7, debian:buster-debian, etc.| 
|`add_to_tags`| `<str>`|tags suffix to recognize given distro: debian, alpine, ami (amazonlinux)| 
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

The yaml validation process includes:

- Ensure defined spec are valid for the manager scope
- Ensure correct tags formatting with keys-values pair
- Ensure no excess keys are being parsed or setup apart from specs (If you want to include new key-value remember to update `specs` before proceeding.)
- Correct data parsing: envars, https, etc.
