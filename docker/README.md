## BentoServer base images

There are four type of BentoServer docker base image:

| Image Type | Description                                | Supported OS                                          | Usage                             |
|------------|--------------------------------------------|-------------------------------------------------------|-----------------------------------|
| `runtime`  | contains latest BentoML releases from PyPI | `debian{11,10}`, `ubi8`, `amazonlinux2`, `alpine3.14` | production ready                  |
| `cudnn`    | runtime + support for CUDA-enabled GPU     | `debian{11,10}`, `ubi8`                               | production ready with GPU support |
| `devel`    | nightly build from development branch      | `debian{11,10}`, `ubi8`                               | for development use only          |
| `conda`    | runtime + conda + optional GPU supports    | `debian{11,10}`,                                      | production ready                  |

* Note: currently there's no nightly devel image with GPU support.

The final docker image tags will have the following format:

```markdown
<release_type>-<python_version>-<distros>-<suffix>-<?:conda>
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

## NOTICE: MISSING PYTHON VERSION ON UBI

Python 3.7 and 3.10 is missing. The reason being RedHat doesn't provide support for these Python version.
If you need to use UBI and Python 3.7 make sure to contact the BentoML team for supports..

## NOTICE: CONDA AVAILABILITY ONLY ON DEBIAN

From 1.0.0a7 onwards, BentoML will only provide conda supports with debian variants only.

We ran into a lot of trouble building BentoML to supports Python from 3.6 to 3.10 with conda environment on other distros than debian. In order to reduce 
complexity we will now only provides conda on Debian-based image. Conda will be available with all of BentoML image type, including `runtime`, `devel`, `cudnn`. 

If you need to use conda on other distros contact the BentoML team for supports.

Example conda tags:
- `bento-server:1.0.0a7-python3.8-debian11-runtime-conda`
- `bento-server:1.0.0a7-python3.8-debian11-cudnn-conda`
- `bento-server:devel-python3.8-debian11-cudnn-conda`
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
```

```shell
docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:base-python${PYTHON_VERSION}-${OS} \
    -f ./generated/bento-server/${OS}/base/Dockerfile \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} .
```

```shell
docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:${BENTOML_VERSION}-python${PYTHON_VERSION}-${OS}-runtime \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} ./generated/bento-server/${OS}/runtime
```

```shell
docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:${BENTOML_VERSION}-python${PYTHON_VERSION}-${OS}-cudnn \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} ./generated/bento-server/${OS}/cudnn
```

```shell
docker buildx build --load ${PLATFORM_ARGS} \
    -t ${IMAGE_NAME}:devel-python${PYTHON_VERSION}-${OS} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} ./generated/bento-server/${OS}/devel
```

FYI: `-v` (Volume mount) and `-u` (User permission) shares directories and files permission between your local machine and Docker container.
Without `-v` your work will be wiped once container exists, where `-u` will have wrong file permission on your host machine.

For CPU-based images:
```shell
# -v and -u are recommended to use.

docker run -it -u $(id -u):$(id -g) -v $(pwd):/bentoml bentoml/bento-server:1.0.0a6-python3.8-ubi8-runtime
```

For GPU-based images:
```shell
docker run --gpus all --device /dev/nvidia0 --device /dev/nvidiactl \
             --device /dev/nvidia-modeset --device /dev/nvidia-uvm \
             --device /dev/nvidia-uvm-tools -i -t -u $(id -u):$(id -g) \
             -v $(pwd):/bentoml bentoml/bento-server:1.0.0a6-python3.8-ubi8-runtime
```


# Development

This section covers how BentoML internally manages its docker base image releases.

## Image Manifest

Dockerfile metadata and related configs for multiple platforms are defined under [manifest/](manifest/) directory.
This can also be used to control releases with CI pipelines. A manifest file is
partially generated with `manager create-manifest`. Each manifest will have
a name format `<package>.cuda_v<cuda_version>.yaml`, since we are supporting GPU
images.

- [common](#common-spec): holds general templates for our template context

### Common Spec

Each of our distros releases will contain the following configs:

| Keys            | Type               | Defintions                                                                                                                           |
| --------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `templates_dir` | `str`              | templates entrypoint, can be found [templates/\*](./templates)                                                                       |
| `base_image`    | `str`              | base distro image: registry.access.redhat.com/ubi8, debian:buster, etc.                                                              |
| `suffixes`      | `str`              | tags suffix to recognize given distro: ubi8, debian11, alpine3.14, ami2 (amazonlinux).                                               |
| `conda`         | `bool`             | Whether to support conda with this distros                                                                                           |
| `ignore_python` | `List[str]`        | List of Python version to be ignored in this distros. This is due to the base distros' lack of support for the given python version. |
| `header`        | `str`              | headers to be included in our Dockerfile. We are using BuildKit + experimental features for Dockerfile to speed up build time.       |
| `envars`        | `List[str]`        | List of environment variables that can be used to build the container images.                                                        |
| `architectures` | `List[str]`        | List of supported architecture type, i.e: arm64, amd64, s390x, etc.                                                                  |
| `release_types` | `List[str]`        | List of supported releases type for BentoML                                                                                          |
| `dependencies`  | `Mapping[str,str]` | a map of required CUDA packages + other required package for given distros                                                           |

**NOTES**: The reason why we named our GPU releases `cudnn` instead of `gpu` is due to clarification. With most deep learning framework
they will have some implementationn of NVIDIA's cuDNN libraries. Thus, with `gpu` naming, end users won't have any indication
of having CUDNN in the images, whereas calling `cudnn` will inform users with enough information about the container itself. We also
provide some others math related library from NVIDIA, which are requirements for most deep learning frameworks. You can always add
additional library like [NCCL](https://developer.nvidia.com/nccl), [TensorRT](https://developer.nvidia.com/tensorrt) via the OS package 
manager with your custom docker image built on top of the one BentoML provided.

## Workflow

High-level workflow when publishing new docker images:

2) generate build context
3) render `j2` templates
4) build from generated directory
5) push to given registries.

We are using `attrs` to validate the data from yaml.
