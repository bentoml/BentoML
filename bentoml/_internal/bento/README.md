# Bento

Bento is a standardized file archive format in BentoML, which describes how to load
and run a `bentoml.Service` defined by a user. It includes user code which instantiates
the `bentoml.Service` instance, as well as related configurations, data/model files,
and dependencies.

## Building a Bento

To build a Bento from your service definition code, simply run the following command
from CLI by providing the path to bentofile.yaml config file:

```bash
bentoml build -f ./bentofile.yaml
```

There is an equivalent Python API for building Bento as well:
```python
import bentoml
bentoml.build(
    'fraud_detector:svc',
    # Other build options
)
```

By default, `build` will include all files in current working directory, besides the
files specified in the `.bentoignore` file in the same directory. It will also automatically
infer all PyPI packages that are required by the service code, and pin down to the version
used in current environment.

User may further customize the build of a service as below:

```python
import bentoml

bentoml.build(
    service="fraud_detector.py:svc",
    version="any_version_label",  # override default version generator
    description=open("README.md").read(),
    include=['*'],
    exclude=[], # files to exclude can also be specified with a .bentoignore file
    additional_models=["iris_model:latest"], # models to pack in Bento, in addition to the models required by service's runners 
    labels={
        "foo": "bar",
        "team": "abc"
    },
    python=dict(
        packages=["tensorflow", "numpy"],
        # requirements_txt="./requirements.txt",
        index_url="http://<api token>:@mycompany.com/pypi/simple",
        trusted_host=["mycompany.com"],
        find_links=['thirdparty..'],
        extra_index_url=["..."],
        pip_args="ANY ADDITIONAL PIP INSTALL ARGS",
        wheels=["./wheels/*"],
    ),
    docker=dict(
        # "base_image": "mycompany.com/registry/name/tag",
        distro="amazonlinux2",
        gpu=True,
        setup_script="setup_docker_container.sh",
        python_version="3.8",
    ),
    conda={
        "environment_yml": "./environment.yml",
        "channels": [],
        "dependencies": [],
    }
)
```


Alternatively, user may put all build options in a `bentofile.yaml` in the same directory
```yaml
service: "iris_classifier:svc"
description: "file: ./readme.md"
labels:
  foo: bar
  team: abc
include:
- "*.py"
- "*.json"
exclude: 
- "*.pyc"
additional_models:
- "iris_model:latest"
docker:
  distro: slim
  gpu: True
  python_version: "3.8"
  setup_script: "./setup_env.sh"
python:
  packages:
    - tensorflow
    - numpy
    - --index-url http://my.package.repo/simple/ SomePackage
    - --extra-index-url http://my.package.repo/simple SomePackage
    - -e ./my_py_lib
  index_url: http://<api token>:@mycompany.com/pypi/simple
  trusted_host: mycompany.com
  # index_url: null # means --no-index
  find_links:
    - file:///local/dir
    - thirdparth...
  extra_index_urls:
    - abc.com
  pip_args: "-- "
  wheels:
    - ./build/my_lib.whl
```

Another bentofile example:
```yaml
service: "foo.bar.another_svc:my_svc"
include:
- "another_svc/**"
docker:
  base_image: "my_own_docker_image:0.1.2"
python:
  requirements_txt: "./requirements.txt"
  index_url: "http://<api token>:@mycompany.com/pypi/simple"
conda:
  channels: ["h2o"]
  dependencies: ["h2o"]
```


When running the CLI command `bentoml build` without any parameter, it will look for
a `bentofile.yaml` file in current directory and use it as build target. User may also
specify which build file to use with the `-f` option, e.g.:

```bash
bentoml build -f ./bentofile-proj2.yaml
```

Note that `version` can not be set via the build yaml file, although user can pass in
a version str when running the build command, e.g.:

```bash
bentoml build --version="dataset_1023_run_3021"
```

The default `build_ctx` will be current directory, it can be changed via CLI arg. In 
this case, BentoML will look for `bentofile.yaml` file in the build context directory
if a build file is not provided.

```bash
bentoml build --build-ctx=~/my_project/
```

## Bento Internals

After a Bento is built, user can find it by using the `bentoml list` and `bentoml get`
CLI command. All bentos created will be stored under `$(USER_HOME)/bentoml/bentos`
directory.

Inside a Bento archive, you will find the following file structure:

```bash
/example_bento
- README.md
- bento.yaml
- /apis
  - openapi.yaml # openapi spec
- /env
  - /python
    - version.txt
      - requirements.txt
      - requirements.lock.txt
      - pip_args.txt
      - /wheels
    - /docker
      - Dockerfile
      - entrypoint.sh
      - init.sh
      - setup_script
    - /conda
      - environment.yml

- /FraudDetector  # this folder is mostly identical to user's development directory
  - bento.py
  - /common
      - my_lib.py
  - my_config.json

- /models
  - /my_nlp_model
    - /zhjw7ssxf3i6zcmf2ie5eubhd
      - bentoml_model.yml
      - model.pkl
    - latest
```

An example `bento.yaml` file in a Bento directory:

```yaml
service: bento:svc
name: fraud_detector
version: 6qc5p2sh4vi6zlj63suqi5nl2
bentoml_version: 1.1.0
created_at: 2021-11-17 20:35:50.968312+00:00
labels:
    foo: bar
    abc: def
    author: parano
    team: bentoml
models:
- my_nlp_model:20210709_C154BA
```


Build docker image from a Bento directory:

```bash
cd bento_path
docker build -f ./env/docker/Dockerfile .
```

