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

bentofile.yaml
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

When the build command is called, BentoML will look at the `bentofile.yaml` for instructions on how to build the bento

.. code-block:: yaml

    # Sample bentofile.yaml
    service: "service.py:svc"  # A convention for locating your service: <YOUR_SERVICE_PY>:<YOUR_SERVICE_ANNOTATION>
    description: "file: ./README.md"
    labels:
        owner: bentoml-team
        stage: demo
    include:
     - "*.py"  # A pattern for matching which files to include in the bento
    python:
      packages:
       - scikit-learn  # Additional libraries to be included in the bento
       - pandas

Built bentos are added the local bento store and can be managed with both Python APIs and CLI.

.. code-block:: bash

    > bentoml list # list all bentos in the store
    > bentoml get iris_classifer:latest # get the description of the bento

The build options by default work for the most common cases but can be further customized by calling 
the `set_build_options()` function on the service. Let's explore the available options. See documentation 
for in-depth details of build options.

Service
^^^^^^^
The `service` parameter is a required field which must specify where the service code is located and under what variable
name the service is instantiated in the code itself, separated by a colon. If either parameters is incorrect, the bento will
not be built properly. BentoML uses this convention to find the service, inspect it and then determine which models should be
packed into the bento.

`<Your Service .py file>:<Variable Name of Service in .py file>`

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

Labels
^^^^^^
The `labels` argument is a key value mapping which sets labels on the bento so that you can add your own custom descriptors to the bento

Include
^^^^^^^

The `include` keyword argument specifies the pathspecs (similar to the .gitignore file) of the Python 
modules and data files to be included in the build. The pathspecs are relative the current working 
directory. If not explicitly specified, all files and directories under the current work directory are 
included in the build.

Try to limit the amount of files that are included in your bento. For example, if unspecified, or if * is specified, all
git versioning in the directory could be included in the bento by accident.

Exclude
^^^^^^^

The `exclude` keyword argument specifies the pathspecs (similar to the .gitignore files) of the Python 
modules or data files to be excluded in the build. The pathspecs are relative the current working 
directory. Users can also opt to place a `.bentoignore` file in the directory where `bentoml build` is 
run to achieve the same file exclusion during build. If not explicitly specified, nothing is excluded 
from the build. Exclude is applied after include.

Environment
^^^^^^^^^^^

The `env` keyword argument specifies the Python version and dependencies required to deploy the bento.
If not explicitly specified, the build to automatically infer the PyPI packages required by the service
by recursively walking through all the dependencies. While the auto-infer in convenient, we still
recommend to define the required package and versions explicitly, to ensure more deterministic build
and deployment.


Additional Models
^^^^^^^^^^^^^^^^^

The build automatically identifies the models and their versions to be built into the bento based on the
:ref:`service definition <service-definition-page>`. The service definition loads runners through
the framework specific `load_runner()` function, the build will identify the model through the tag
provided in the arguments. Use the `additional_models`` keyword argument to include models tags that
are used in customer `runners`.


Docker Options
^^^^^^^^^^^^^^

The `docker` Options for generating the Docker image of the Bento service, such as selecting the base image and
enabling the use of GPU.

+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Field           | Description                                                                                                        |
+=================+====================================================================================================================+
| distro          | Configure the particular os distribution on the Docker image ["debian", "amazonlinux2", "alpine", "ubi8", "ubi7"]  |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| python_version  | Specify which python to include on the Docker image ["3.7", "3.8", "3.9"]                                          |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| gpu             | Determine if your container will have a gpu. This is not compatible with certain distros                           |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| devel           | If you want to use the latest main branch from the BentoML repo in your bento                                      |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| setup_script    | Is a python or shell script that executes during docker build time                                                 |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| base_image      | Is a user-provided custom docker base image. This will override all other custom attributes of the image           |
+-----------------+--------------------------------------------------------------------------------------------------------------------+

Python Options
^^^^^^^^^^^^^^

The `python` Options to be included in the bento's environment can be specified here. Generally BentoML will infer most of
these options for you, but if there's anything extra that you'd like to include you can specify it here

+-------------------+------------------------------------------------------------------------------------+
| Field             | Description                                                                        |
+===================+====================================================================================+
| requirements_txt  | The path to a custom requirements.txt file                                         |
+-------------------+------------------------------------------------------------------------------------+
| packages          | Packages to include in this bento                                                  |
+-------------------+------------------------------------------------------------------------------------+
| lock_packages     | Whether to lock the packages or not                                                |
+-------------------+------------------------------------------------------------------------------------+
| index_url         | Inputs for the `--index-url` pip argument                                          |
+-------------------+------------------------------------------------------------------------------------+
| no_index          | Whether to include the `--no-index` pip argument                                   |
+-------------------+------------------------------------------------------------------------------------+
| trusted_host      | List of trusted hosts used as inputs using the `--trusted-host` pip argument       |
+-------------------+------------------------------------------------------------------------------------+
| find_links        | List of links to find as inputs using the `--find-links` pip argument              |
+-------------------+------------------------------------------------------------------------------------+
| extra_index_url   | List of extra index urls as inputs using the `≈` pip argument                      |
+-------------------+------------------------------------------------------------------------------------+
| pip_args          | Any additional pip arguments that you would like to add when installing a package  |
+-------------------+------------------------------------------------------------------------------------+
| wheels            | List of paths to wheels to include in the bento                                    |
+-------------------+------------------------------------------------------------------------------------+

Conda Options
^^^^^^^^^^^^^

The `conda` options are to configure the conda environment which the bento will run.

+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| Field            | Description                                                                                                                      |
+==================+==================================================================================================================================+
| environment_yml  | Path to a conda environment file to copy into the bento. If specified, this file will overwrite any additional option specified  |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| channels         | Custom conda channels to use. If not specified will use "defaults"                                                               |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| dependencies     | Custom conda dependencies to include in the environment                                                                          |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| pip              | The specific "pip" conda dependencies to include                                                                                 |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+

