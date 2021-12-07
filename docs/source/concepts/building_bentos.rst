.. _building-bentos-page:

Building Bentos
***************

Bentos are the distribution format for BentoML services. A bento is a self-contained archive that 
includes all the information required, such as models, code, configurations, and data files, for 
deploying a service.

Bento File Structure
--------------------

Bento follows the directory structure below. We will look into each file and directory in more details.

.. code-block:: bash

    > cd $(bentoml get iris_classifier:latest --path)
    > ls -lR
    - README.md
    - bento.yaml
    - /apis/
        - openapi.yaml
    - /env/
        - /python
            - python_version.txt
            - requirements.txt
            - /wheels
        - /docker
            - Dockerfile
            - docker-entrypoint.sh
            - bentoml-init.sh
        - /conda
            - environment.yml
    - /src
        - bento.py
        - /common
            - my_lib.py
        - my_config.json
    - /models
        - /iris_classifier
            - /yftvuwkbbbi6zcphca6rzl235
                - model.pkl
                - bentoml_model.yaml 

README.md
^^^^^^^^^

The README file in markdown format that describes this Bento service.

bento.yaml
^^^^^^^^^^

Configuration of the Bento service.

/apis Directory
^^^^^^^^^^^^^^^

The directory that contains the API specifications of the Bento service. The OpenAPI specifications can 
be found in openapi.json. Support for Protobuf is under the current roadmap and will be supported in a 
future version.

/env Directory
^^^^^^^^^^^^^^

The configurations of the dependent environments. Python versions and requirements can be found under 
`/python`. Docker files and entry point shell scripts can be found under `/docker`. Conda environment 
configuration can be found under `/conda`.

/svc Directory
^^^^^^^^^^^^^^

The directory that includes the service definition and its dependent modules. Data files required by 
the service are also found under this directory.

/models Directory
^^^^^^^^^^^^^^^^^

The directory that includes the dependent models of the service definition.

Build Options
-------------

Recall the :ref:`Getting Started <getting-started-page>` guide, bentos are built with the `build` CLI 
command.

.. code-block:: bash

    > bentoml build ./bento.py:svc

Built bentos are added the local bento store and can be managed with both Python APIs and CLI.

.. code-block:: bash

    > bentoml list # list all bentos in the store
    > bentoml get iris_classifer:latest # get the description of the bento

The build options by default work for the most common cases but can be further customized by calling 
the `set_build_options()` function on the service. Let's explore the available options. See documentation 
for in-depth details of build options.

Version
^^^^^^^

The version of the bento to be built can be specified by the `bento` keyword argument. If not explicitly 
specified, the version is automatically generated based on the timestamp of the build combined with random bytes.

Description
^^^^^^^^^^^

The keyword argument sets the `description` of the Bento service. The contents will be used to create the 
`README.md` file in the bento archive. If not explicitly specified, the build to first look for the 
presence of a `README.md` in the current working directory and set the contents of the file as the 
description.

Include
^^^^^^^

The `include` keyword argument specifies the pathspecs (similar to the .gitignore file) of the Python 
modules and data files to be included in the build. The pathspecs are relative the current working 
directory. If not explicitly specified, all files and directories under the current work directory are 
included in the build.

Exclude
^^^^^^^

The `exclude` keyword argument specifies the pathspecs (similar to the .gitignore files) of the Python 
modules or data files to be excluded in the build. The pathspecs are relative the current working 
directory. Users can also opt to place a `.bentoignore` file in the directory where `bentoml build` is 
run to achieve the same file exclusion during build. If not explicitly specified, nothing is excluded 
from the build. Exclude is applied after include.

Docker Options
^^^^^^^^^^^^^^

Options for generating the Docker image of the Bento service, such as selecting the base image and 
enabling the use of GPU.

.. todo::

    Add Docker options examples


Environment
^^^^^^^^^^^

The `env` keyword argument specifies the Python version and dependencies required to deploy the bento. 
If not explicitly specified, the build to automatically infer the PyPI packages required by the service 
by recursively walking through all the dependencies. While the auto-infer in convenient, we still 
recommend to define the required package and versions explicitly, to ensure more deterministic build 
and deployment.

.. todo::

    Add Environment options examples

Additional Models
^^^^^^^^^^^^^^^^^

The build automatically identifies the models and their versions to be built into the bento based on the 
:ref:`service definition <service-definition-page>`. The the service definition loads runners through 
the framework specific `load_runner()` function, the build will identify the model through the tag 
provided in the arguments. Use the `additional_models`` keyword argument to include models tags that 
are used in customer `runners`.

.. todo::

    Add the further reading section

