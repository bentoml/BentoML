"""
Bento is a standardized file archive format in BentoML, which describes how to setup
and run a `bentoml.Service` defined by a user. It includes user code which instantiates
the `bentoml.Service` instance, as well as related configurations, data/model files,
and dependencies.

Here's an example file structure of a Bento

    /example_bento
     - readme.md
     - bento.yml
     - /api/
         - swagger.json # swagger definition
         - proto.pb # Note: gRPC proto is not currently available
     - /env/
         - /python
             - python_version.txt
             - requirements.txt
             - /wheels
         - /docker
             - Dockerfile
           - Dockerfile-gpu
           - Dockerfile-foo
             - docker-entrypoint.sh
           - bentoml-init.sh
           - setup-script # optional
         - /conda
             - environment.yml

     - /FraudDetector  # this folder should be pretty much identical to user's development project dir
        - bento.py
        - /common
           - my_lib.py
        - my_config.json
            - /my_nlp_model
                 - bentoml_model.yml
                 - model.pkl

cd bundle_path
docker build -f ./env/docker/Dockerfile .
"""


# TODO:
def load():
    pass


# TODO:
def containerize():
    pass
