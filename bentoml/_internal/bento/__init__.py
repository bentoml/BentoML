"""
Bento is a standardized file archive format in BentoML, which describes how to setup
and run a `bentoml.Service` defined by a user. It includes user code which instantiates
the `bentoml.Service` instance, as well as related configurations, data/model files,
and dependencies.

Here's an example file structure of a Bento

    /example_bento
     - readme.md
     - bento.yml
     - /apis/
         - openapi.yaml # openapi spec
         - proto.pb # Note: gRPC proto is not currently available
     - /env/
         - /python
             - python_version.txt
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

An example `bento.yml` file in a Bento directory:

    service: bento.py:svc
    name: FraudDetector
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



Build docker image from a Bento directory:

    cd bento_path
    docker build -f ./env/docker/Dockerfile .
"""
