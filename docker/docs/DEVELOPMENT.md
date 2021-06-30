## BentoML Container Generation Workflow Documentation

We will go into how BentoML manages its docker releases.

1. [Image Manifests](#image-manifest) explains the YAML file that control our Docker releases.
1. [Workflow](#workflow) demonstrates scripts' logics and its implementation.
1. [CI](#ci) will go into how we releases nightly/devel images.
1. [Usecase and Troubleshooting](#usecase-and-troubleshooting) enables debugging and quickly development.

### Notes

#### Development environment

Assume that you have already set up the development environment correctly. Refers to main README's [Developing section](https://github.com/bentoml/BentoML/tree/master/docker#developing).

If you prefer to run the environment locally without using `docker`, then

1. Install [`poetry`](https://python-poetry.org/docs/#installation)

1. ```poetry install``` to install required dependencies

1. enters virtualenv with ```poetry shell```

1. Once in the virtualenv, run:

```shell
» ./manager.py --helpfull
```

#### Ephemeral directory

_meaning you shouldn't edit anything within the directory_

- [```model-server```](./model-server)
- [```yatai-service```](./yatai-service)
- [```generated```](../generated)

### Image Manifest

Dockerfile metadata and related configs for multiple platforms are defined under [manifest.yml](../manifest.yml).
In the future, this file will be distributed into a *manifest directory* and would be used to control CI pipelines.

The manifest file will be validated while invoking `manager.py`. Refers to [Validation](#validation) for implementations.

#### Manifest structure
- [repository](#repository): enable supports for multiple registries: Heroku, Docker Hub, GCR, NVCR.
- [cuda_dependencies](#cuda_dependencies): provides options for multiple CUDA releases.
- [release_spec](#release_spec): determines releases template for every supported OS.
- [packages](#packages): determines supported OS releases for each BentoML's package.
- [releases](#releases): detailed instruction for each flavor of Linux distros.

##### repository

To determine new registry:
```yaml
docker.io:
  only_if: HUB_PASSWORD
  user: HUB_USERNAME
  pwd: HUB_PASSWORD
  registry:
    model-server: docker.io/bentoml/model-server
    yatai-service: docker.io/bentoml/yatai-service
```

A registry definition allows us to set up correct docker information to push our final images, tests, and scan.

- `repository_name`: docker.io, gcr.io, ecr.io, etc.

| Keys | Type | defintions |
|------|------|------------|
|`only_if`| `<str>`|Check if password has been set as envars, otherwise skip given registry| 
|`user`| `<str>`| envars for registry username| 
|`pwd`| `<bool>`|envars for registry password| 
|`registry`| `<dict>`|contains keys as docker package and values as target registry URL| 

##### cuda_dependencies

To determine new CUDA version:
```yaml
11.3.1: &cuda11_3_1
  cudart: 11.3.109-1
  cudnn8: 8.2.0.53-1
  libcublas: 11.5.1.109-1
  libcurand: 10.2.4.109-1
  libcusparse: 11.6.0.109-1
  libcufft: 10.4.2.109-1
  libcusolver: 11.1.2.109-1
```

This allows us to add multiple versions of CUDA to support for current DL frameworks as well as backward compatibility

CUDA semver are also checked for validity. Usually devs should use this key-value pairs as this 
are required library for CUDA and cuDNN. Refers to [cuda's repos](https://developer.download.nvidia.com/compute/cuda/repos/) 
for supported distros and library version.

##### release_spec

Each of our distros releases will follow the following:
```yaml
templates_dir: ""
base_image: ""
add_to_tags: ""
multistage_image: True
header: |
  # syntax = docker/dockerfile:1.2
  #
  # ===========================================
  #
  # THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
  #
  # ===========================================
envars:
  - LANG=C.UTF-8
  - LC_ALL=C.UTF-8
  - PYTHONDONTWRITEBYTECODE=1
cuda_prefix_url: ""
cuda_requires: "cuda>=11.3 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"
cuda:
  <<: *cuda11_3_1 # cuda ref the previous segment
```

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
|`cuda`| `<dict[str,str]>`|list of CUDA components, refers to [CUDA components](#cuda_dependencies)| 

##### packages

This is how we define Linux distros for each packages.
```yaml
model-server:
  devel: &default_specs
    - debian
    - centos
  runtime:
    - *default_specs
    - amazonlinux
    - alpine
  cudnn:
    - *default_specs
yatai-service:
  runtime:
    - *default_specs
```

##### releases

All the specifics for each releases based on [release_spec](#release_spec):
```yaml
debian:
  10:
    <<: *release_spec
    templates_dir: templates/debian
    base_image: debian:buster-slim
    add_to_tags: "slim"
    cuda_prefix_url: "ubuntu2004"
```

### Workflow

*UNDER CONSTRUCTION*

##### Validation

*UNDER CONSTRUCTION*

### CI

*UNDER CONSTRUCTION*

### Usecase and Troubleshooting

*UNDER CONSTRUCTION*