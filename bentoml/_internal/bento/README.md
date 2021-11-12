 # Bento

Bento is a standardized file archive format in BentoML, which describes how to load
and run a `bentoml.Service` defined by a user. It includes user code which instantiates
the `bentoml.Service` instance, as well as related configurations, data/model files,
and dependencies.

Here's an example file structure of a Bento

```bash
/example_bento
 - README.md
 - bento.yaml
 - /apis
     - openapi.yaml # openapi spec
     - proto.pb # Note: gRPC proto is not currently available
 - /env
     - /python
         - version.txt
         - requirements.txt
         - /wheels
     - /docker
         - Dockerfile
         - Dockerfile-gpu  # optional
         - Dockerfile-foo  # optional
         - docker-entrypoint.sh
         - bentoml-init.sh
         - setup-script  # optional
     - /conda
         - environment.yml

 - /FraudDetector  # this folder is mostly identical to user's development directory
    - bento.py
    - /common
       - my_lib.py
    - my_config.json

 - /models
    - /my_nlp_model
       - bentoml_model.yml
       - model.pkl
```

An example `bento.yaml` file in a Bento directory:

```yaml
service: bento:svc
name: fraud_detector
version: 20210709_DE14C9
bentoml_version: 1.1.0

created_at: ...

labels:
    foo: bar
    abc: def
    author: parano
    team: bentoml

apis:
- predict:  # api_name is the key
    route: ...
    docs: ...
    input:
        type: bentoml.io.PandasDataFrame
        options:
            orient: "column"
    output:
        type: bentoml.io.JSON
        options:
            schema:
                # pydantic model .schema() here

models:
- my_nlp_model:20210709_C154BA
```



Build docker image from a Bento directory:

```bash
cd bento_path
docker build -f ./env/docker/Dockerfile .
```


User may customize build options for a service:

```python
import bentoml

svc = bentoml.Service(__name__)

svc.set_build_options(
    version="any_version_label",
    description=open("README.md").read(),
    models=["iris_model:latest"],
    include=['*'],
    exclude=[], # files to exclude can also be specified with a .bentoignore file
    labels={
        "foo": "bar",
    },
    env={
        "python": {
            "version": '3.7.11',
            "wheels": ["./wheels/*"],
            "pip_install": "auto",
            # "pip_install": "./requirements.txt",
            # "pip_install": ["tensorflow", "numpy"],
            # "pip_install": bentoml.build_utils.lock_pypi_version(["tensorflow", "numpy"]),
            # "pip_install": None, # do not install any python packages automatically
        },
        "docker": {
            # "base_image": "mycompany.com/registry/name/tag",
            "distro": "amazonlinux2",
            "gpu": True,
            "setup_script": "sh setup_docker_container.sh",
        },
        # "conda": "./environment.yml",
        "conda": {
            "channels": [],
            "dependencies": []
        }
    },
)
```