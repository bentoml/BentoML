.. _bento-format-page:

Building Bentos
===============

Bento is a standardized file archive format in BentoML that describes
how to load and run a ``bentoml.Service`` defined by the user. It includes
code that instantiates the ``bentoml.Service`` instance, as well
as related configurations, data/model files, and dependencies.

Let's build a Bento
===================

Some common situations for having a custom Bento file are if you already
have a 

First, let's customize the ``.bentoignore``
-------------------------------------------

We don't need __pycache__/ files, so let's add that along with
other unnecessary folders and files. The format is identical to how
a ``.gitignore`` would function.

.. note::

   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints

Bento Format
================

BentoML is a standard file format that describes how to load and run
a ``bentoml.Service`` defined by the user. It includes code that
instantiates the ``bentoml.Service`` instance, as well as related
configurations, data/model files, and dependencies.

BentoML file format is a YAML file that looks like

.. code:: yaml

   service: "service:svc"
   description: "file: ./README.md"
   labels:
      owner: bentoml-team
      stage: demo
   include:
      - "*.py"
   python:
      packages:
         - scikit-learn
         - pandas

Python Packages
===============

Whether you're using pip or conda, you can specify which Python packages
to include in your Bento by configuring them in ``bentofile.yaml``.

Python Options
--------------

Python Options are used to customize the behavior of the Bento
and how BentoML sets up the Docker image.

If you're using an existing python environment, you can use a
``requirements.txt``. For a project, you can run
``pip freeze > requirements.txt`` to generate a requirements file
to load with BentoML.

.. code:: yaml

   python:
     requirements_txt: "requirements.txt"


When you're creating a new project, you can also specify Python packages
straight from the Bento file.

The ``packages`` key is used when there isn't an existing requirements.txt
file. You can list packages and specify their version as well.

.. code:: yaml

   python:
     packages:
        - numpy
        - "matplotlib==3.5.1"

Additionally, there are many keys that can help manage larger projects.

.. code:: yaml

   python:
     requirements_txt: "requirements.txt"
     lock_packages: True
     index_url: "https://example.org/"
     no_index: False
     trusted_host: "example.org"
     find_links:
        - "https://test.org/"
     extra_index_url:
        - "https://test.org/"
     pip_args: "--quiet"
     wheels:
        - "https://example.org/wheels/packages.whl"

If you're using a pip wheel, you can include a local or external link
to it under the ``wheels`` key.

For serving, you can also specify trusted hosts

Conda Options
-------------

Similarly to PyPi, you can use Conda to handle dependencies.
By running ``conda export``, you can generate an ``environment.yml``
to use.

.. code:: yaml

   conda:
     environment_yml: "environment.yml"

And in the same vein, you can always specify the ``dependencies`` key instead
of using conda export.

.. code:: yaml

   conda:
     dependencies:
        - "scikit-learn==1.2.0"
        - numpy

If some of the dependencies are from different conda channels, the Bento file
can also handle that with

.. code:: yaml

   conda:
     channels:
        - "conda-forge"
        - bioconda
        - r

Docker Options
--------------

BentoML makes it easy to deploy a Bento to a Docker container.

Here's a basic Docker option key.

.. code:: yaml

   docker:
     distro: debian
     gpu: True
     python_version: "3.8.9"
     setup_script: "setup.sh"

For the ``distro`` options, you can choose from 5.

- debian
- amazonlinux2
- alpine
- ubi8
- ubi7

This config can be explored from `BentoML's Docker page <https://hub.docker.com/r/bentoml/bento-server>`_.

The gpu key will also allocate a GPU in the Docker. If you're using the standard devices variable in PyTorch,
for example, this key will enable the gpu.

For more interesting docker development, you can also use a ``setup.sh`` for the container.
If you're using debian, you can do something like this:

.. code:: shell

   sudo apt update && sudo apt install software-properties-common
   sudo add-apt-repository 'deb [arch=amd64] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.0 multiverse'
   sudo apt update && sudo apt install mongodb-org
   sudo apt upgrade

Building a Bento
================

Let's now create a ``bentofile.yaml`` file for generating
the Bento.

.. code:: yaml

   service: "service:svc"
   description: "file: ./README.md"
   labels:
      owner: bentoml-team
      stage: demo
   include:
      - "*.py"
   python:
      packages:
         - scikit-learn
         - pandas

To build a Bento from your service definition code, simply run the
following command from CLI and provide the path to bentofile.yaml
config file:

.. code:: bash

   bentoml build -f ./bentofile.yaml

By default, ``build`` will include all files in current working
directory, besides the files specified in the ``.bentoignore`` file in
the same directory. It will also automatically infer all PyPI packages
that are required by the service code, and pin down to the version used
in current environment.

For larger projects, the user may need more customization.
The Bento format has a variety of options, here is a
``bentofile.yaml`` file as an example:

.. code:: yaml

   service: "service:svc"
   description: "file: ./README.md"
   labels:
     foo: bar
   include:
     - "*.py"
     - "*.json"
   exclude:
     - "*.pyc"
   additional_models:
     - "iris_model:latest"
   conda:
     dependencies:
        - "scikit-learn==1.2.0"
        - numpy
     channels:
        - "conda-forge"
        - bioconda
        - r
   docker:
     distro: debian
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
     lock_packages: True
