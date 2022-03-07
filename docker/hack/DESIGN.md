`manager --help`

## goals

- Support multiple platforms (arm64, arm32, ppc64le) as possible 
- multiple OS (UBI, alpine, debian)
- multiple Python version 
- supports for multiple different docker registries (Quay, GCR, Docker.io)

This tools should be extensible, meaning users can use this to quickly create
any work on any specs to manage any given docker registry.

- Can be used in CI : GitHub CI <:- TODO

## requirements

We will utilize `docker buildx` underthe hood, so make sure to have that installed with `docker buildx install` (buildx should already installed for folks with Docker > 19.03)

we will also use [docker-pushrm](https://github.com/christian-korneck/docker-pushrm) to push the given readme, so make sure to have that installed as well.

in order to setup for multiple architecture [QEMU](https://github.com/multiarch/qemu-user-static) is required. Install with:
```bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

## workflow

generate all required files with:
```bash
manager generate bento-server --target_version 1.0.0a5
```

under the hood, will create generation and build context whenever `generate` is
called, this will be saved to a json files if `--save_metadata` is passed.
if `--authenticate` is passed, will try to authenticate all defined registry
under the given yaml spec under `$BENTOML_GIT_ROOT/<docker_registry>.yml`

<br>

then we need to authenticate into the given docker registry. Makesure to saved
a `.env` under `$BENTOML_GIT_ROOT/.env` for all the required fields for the
given registry:

```bash
manager authenticate bento-server --registry docker.io

manager authenticate bento-server --registry quay.io
```

This will login into the respected registry for the given docker images. The
follow table demonstrate the ENVARS will be used for each different registry:

| Registry | ENVARS | CLI commands | Notes |
|----------|--------|--------------|-------|
| Docker   |`DOCKER_USER` & `DOCKER_PASSWORD`| `docker login` | |
| Quay.io  |`QUAY_USER` & `QUAY_PASSWORD` | `docker login quay.io` | [(_how to use Quay registry_)](https://github.com/christian-korneck/docker-pushrm#log-in-to-quay-registry) |
| Harborv2 |`HARBOR_USER` &`HARBOR_PASSWORD` | `docker login demo.goharbor.io` | |
| GCR      | TODO | | |
| ECR      | TODO | | |


This will outputs the given generated files to `$BENTOML_GIT_ROOT/docker/generated` with the following file structure:

- `base` directory -> this includes all the basic tools + dependencies that is required for a docker images
- `runtime` directory -> serves as the main entry point for a docker container releases
- `cudnn` directory -> enable GPU support for the package
- `devel` directory -> running the given library installation from git
- `devel-trt` directory -> this will install the library nightly and also add additional TensorRT supports (builds on top of cudnn images)

The hierachy is internally managed by `manager`


```bash
generated
└── bento-server
    ├── README.md
    ├── versions # contains the versions for this specific generation
    ├── alpine3.14
    │   ├── base
    │   │   └── Dockerfile
    │   └── runtime
    │       └── Dockerfile
    ├── amazonlinux2
    │   ├── base
    │   │   └── Dockerfile
    │   └── runtime
    │       └── Dockerfile
    ├── debian11
    │   ├── base
    │   │   └── Dockerfile
    │   ├── cudnn
    │   │   └── Dockerfile
    │   ├── devel
    │   │   └── Dockerfile
    │   ├── devel-trt
    │   │   └── Dockerfile
    │   └── runtime
    │       └── Dockerfile
    └── ubi8
        ├── base
        │   └── Dockerfile
        ├── cudnn
        │   ├── Dockerfile
        │   ├── cuda.repo-amd64
        │   ├── cuda.repo-arm64v8
        │   ├── cuda.repo-ppc64le
        │   ├── nvidia-ml.repo-amd64
        │   ├── nvidia-ml.repo-arm64v8
        │   └── nvidia-ml.repo-ppc64le
        ├── devel
        │   └── Dockerfile
        └── runtime
            └── Dockerfile
```

Example image tags spec:
- `bento-server:devel-python3.7-debian`
- `bento-server:1.0.0-python3.8-ubi8-cudnn`
- `bento-server:1.0.0-python3.7-ami2-runtime`

To manually build any specific distros with support for multiple architecture:
```bash
cd generated/bento-server
VERSION=$(cat versions)
PYTHON_VERSION='3.9'
DISTROS='debian11'
RELEASES='cudnn'

# need to build base images first before anything:
docker buildx build --push --platform linux/amd64,linux/arm64,linux/ppc64le --build-arg PYTHON_VERSION=${PYTHON_VERSION} -t bento-server:base-python${PYTHON_VERSION}-${DISTROS} ${DISTROS}/base

# then build the targetted images:
docker buildx build --push --platform linux/amd64,linux/arm64,linux/ppc64le --build-arg PYTHON_VERSION=${PYTHON_VERSION} -t bento-server:${VERSION}-python${PYTHON_VERSION}-${DISTROS}-${RELEASES} ${DISTROS}/${RELEASES}
```

NOTE: for `docker buildx` we need to manage the generated manifest for multiple
platform and `--push` will save the the images locally or push directly to the
docker registry. `manager` will use `--push` by default, more on this later.

or use manager:
```bash
manager build bento-server --releases cudnn --python_version 3.9 --distros debian11
```
This will only build the images using one single instances. 

to build every images and releases possible generated from specs and push:
```bash
manager build bento-server --all
```
This will utilize a threadpoolexecutor managed by the specs specification, default to use max 5 workers

## blockers
- generate multiple buildx builder?
- builder to be stateless


<details>

<summary>TODO:</summary>
<br>

create a `manifest/<docker-registry>.yaml` with defined releases ordered by
hierachy (separated by ,):
- `manager generate --yaml bento-server --releases runtime,cudnn(gpu_supported),devel`

</details>


## specs

```bash
manager authenticate bento-server --registry docker.io

manager authenticate bento-server --registry quay.io
```
Authenticate registry from envars.

```python
def create_context(repo_manifest: PathLike, validation: bool):
```
Creates context from given YAML file

Input:
    - repo_manifest: YAML spec for this docker registry, located $GIT_ROOT/docker/manifest/<docker_registry>.yml

Options:
    - --validate: validation using cerberus

Default specs will be under $GIT_ROOT/docker/manager/specs.yml

manager create_context --validate bento-server
manager create_context bento-server
-> output two context: 
- dockerfiles.json: which will be used for generating dockerfiles
- images.json: which will be used for generating images

```bash
manager generate <OPTIONS>
```

templates will located under $GIT_ROOT/docker/templates/
templates will only accepts jinja files
- structures for templates
metadata will retrieved from generated directory outputed from `create_context`


manager generate --dockerfiles
manager generate --docs
manager generate --all <:- default to generate both docs and dockerfiles

manager generate --yaml <docker_registry> <:- TODO




```bash
manager build <OPTIONS>
```

build from generated Dockerfiles
