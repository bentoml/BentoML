# BentoML Container Generation Workflow Documentation

We will go into how BentoML manages its docker releases.

1. [Image Manifests](#image-manifest) explains the YAML file that control our Docker releases.
1. [Workflow](#workflow) demonstrates scripts' logics and its implementation.

## Notes

Assume that you have already set up the development environment correctly. Refers to main README's [Developing section](https://github.com/bentoml/BentoML/tree/master/docker#developing).

### Ephemeral directory

_tip: don't edit_

- [```model-server```](./model-server)
- [```yatai-service```](./yatai-service)
- [```generated```](../generated)

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

### specs
We will define how our `manifest.yml` structure under `specs`, and others sections on the file can reference to each of sub keys.
You can see more [here](../manifest.yml).

#### repository

To determine new registry:
```yaml
repository: &repository_spec
  user: HUB_USERNAME
  pwd: HUB_PASSWORD
  urls:
    api: https://hub.docker.com/v2/users/login/
    repos: https://hub.docker.com/v2/repositories
  registry:
    model-server: docker.io/bentoml/model-server
    yatai-service: docker.io/bentoml/yatai-service
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
  <<: *cuda_spec # cuda ref the previous segment
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
|`cuda`| `<dict[str,str]>`|list of CUDA components, refers to [CUDA components](#dependencies)| 

#### tag

We will define our tag validation as follows:
```yaml
fmt: "{release_type}-python{python_version}-{suffixes}"
release_type:
python_version:
suffixes:
```

| Keys | Type | defintions |
|------|------|------------|
|`fmt`| `<str>`|specific python3 string format| 
|`release_type`| `<str>`|devel - runtime - cudnn| 
|`python_version`| `<str>`|3.6 -3.7 - 3.8| 
|`suffixes`| `<str>`|addtional suffix needed to add| 

**NOTES**: The reason why we named our GPU releases `cudnn` instead of `gpu` is due to clarification. With most deep learning framework
they will have some implementationn of NVIDIA's cuDNN libraries. Thus, with `gpu` naming, end users won't have any indication
of having CUDNN in the images, whereas calling `cudnn` will inform users with enough information about the container itself. We also
provide some others math related library from NVIDIA, which are requirements for most deep learning frameworks. You can always add
additional library like [NCCL](https://developer.nvidia.com/nccl), [TensorRT](https://developer.nvidia.com/tensorrt) via the OS package 
manager with your custom docker image built on top of the one BentoML provided.

Tag will also be validated whether keys are defined within the format scope.

### packages

This is how we define Linux distros for each packages.
```yaml
model-server:
  devel: &default_specs
    - debian10
    - centos7
    - centos8
  runtime:
    - *default_specs
    - amazonlinux2
    - alpine3.14
  cudnn:
    - *default_specs
yatai-service:
  runtime:
    - *default_specs
```

**NOTES**: We will define distros as <distro_releases><distro_version> to simplify the parsing process in Python.

### distros

All the specifics for each releases based on [release](#releases):
```yaml
debian10:
  <<: *tmpl_spec
  templates_dir: templates/debian
  base_image: debian:buster-slim
  add_to_tags: "slim"
  cuda_prefix_url: "ubuntu2004"
```

## Workflow

  validate yaml file &rarr; generate build context &rarr; render `j2` templates &rarr; build from generated directory &rarr; Push to given registries.


Each processed will be managed by [``ManagerClient``](../manager.py#L617) with provided functions via `Mixin` class.

(there are still a lot more improvement, so contributions are greatly appreciated.)

We are using [`cerberus`](https://docs.python-cerberus.org/en/stable/index.html) to validate our schema. [MetadataSpecValidator](../utils.py#L267) will handle all
of our validation logics. It ensures:

- defined spec are valid for the manager scope.
- Ensure correct tags formatting with keys-values pair.
- Ensure no excess keys are being parsed or setup apart from specs (If you want to include new key-value remember to update `specs` before proceeding.)
- Correct data parsing: envars, https, etc.