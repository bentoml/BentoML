.. _bento-format-page:

Building Bentos
===============

Bento is a standardized file archive format in BentoML that describes
how to load and run a ``bentoml.Service`` defined by the user. It includes
code that instantiates the ``bentoml.Service`` instance, as well
as related configurations, data/model files, and dependencies.

Let's build a Bento for a gallery project
=========================================

To follow along, clone the `bentoml/gallery/quickstart project
<https://github.com/bentoml/gallery/>`_.

Customizing the ``.bentoignore``
--------------------------------

We don't need __pycache__/ files, so let's add that along with
other unnecessary folders and files. The format is identical to how
a ``.gitignore`` would function.

.. note::
   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints

Building a Bento
----------------

Let's now create a ``bentofile.yaml`` file for generating
the bento. This file is used to give directions for deploying
the quickstart project.

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

The ``service`` field is required and specifies the name of the service
to be loaded from the ``bentoml.Service`` class.

The ``include`` field is optional and specifies the files to be included
in the Bento. If not specified, all files in the current working directory
will be included.

The ``labels`` field is optional and specifies labels for the Bento.

The ``python`` field is optional and specifies the Python packages to be
included in the Bento.

The ``description`` field is optional and specifies the description of
the Bento.

The ``bento.yaml`` file is optional and specifies the Docker configuration
for the Bento.

The ``bentofile.yaml`` file is optional and specifies the BentoML file
format for the Bento.

The ``bentofile.yaml`` file is optional and specifies the BentoML file
format for the Bentig

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
     lock_packages: true

Python Packages
===============

Whether you're using pip or conda, you can specify which Python packages
to include in your Bento by configuring them in ``bentofile.yaml``.

Python Options
--------------

Python Options are used to customize the behavior of the Bento
and how BentoML sets up the Docker image.

.. code:: yaml

   python:
     requirements_txt: "requirements.txt"
     packages:
        - numpy
     lock_packages:
        - scikit-learn==1.2.0
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

Conda Options
-------------

Similarly to PyPi, you can use Conda to handle dependencies.

.. code:: yaml

   conda:
     environment_yml: "environment.yml"
     channels:
        - "conda-forge"
     dependencies:
        - "scikit-learn==1.2.0"
        - numpy
     pip:
        - pytorch

Docker Options
--------------

BentoML makes it easy to deploy a Bento to a Docker container.

.. code:: yaml

   docker:
     distro: debian
     gpu: True
     python_version: "3.8.9"
     setup_script: "setup.sh"
