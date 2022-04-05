.. \_bento-format-page:

# Building Bentos

Bento is a standardized file archive format in BentoML that describes
how to load and run a `bentoml.Service` defined by the user. It includes
code that instantiates the `bentoml.Service` instance, as well
as related configurations, data/model files, and dependencies.

A Bento can be built with the `bentoml build` command with the `bentofile.yaml`
configuration file. Here's an example of that process from the `quickstart guide (https://github.com/bentoml/gallery/tree/main/quickstart#build-bento-for-deployment)_`.

.. code:: yaml

service: "service:svc"
description: "file: ./README.md"
labels:
owner: bentoml-team
stage: demo
include: - "\*.py"
python:
packages: - scikit-learn - pandas

The service field is the python module that holds the bentoml.Service
instance.

## Configuring files to include

In the example above, the `*.py` is including every Python file in
the working directory.

You can also include other wildcard and directory matching.

.. code:: yaml

...

include: - "data/" - "\*_/_.py" - "config/\*.json"

If the include field is not specified, BentoML, by default, will include
every file in the working directory.

If the user needs to include a lot of files, another approach is to
only specify which files to be ignored.

In this situation, the user can use a `.bentoignore` file by placing it
in the working directory and all the files specified there will be ignored
when building the Bento.

This is what a `.bentoignore` file would look like.

.. note::

   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints/
   training_data/

To build a Bento, simply run the following command from your project
directory that contains your `bentofile.yaml`:

.. code:: bash

bentoml build

By default, `build` will include all files in current working
directory, besides the files specified in the `.bentoignore` file in
the same directory. It will also automatically infer all PyPI packages
that are required by the service code, and pin down the version used
in current environment.

Bento Format
============

BentoML is a standard file format that describes how to load and run
a `bentoml.Service` defined by the user. It includes code that
instantiates the `bentoml.Service` instance, as well as related
configurations, data/model files, and dependencies.

.. code:: yaml

service: "service:svc"
description: "file: ./README.md"
labels:
owner: bentoml-team
stage: demo
include: - "\*.py"
python:
packages: - scikit-learn - pandas

# Python Packages

Whether you're using pip or conda, you can specify which Python packages
to include in your Bento by configuring them in `bentofile.yaml`.

## Python Options

There are two ways to specify packages in the Bentofile. First,
we can list packages like below. When left without a version,
pip will just use the latest release.

.. code:: yaml

python:
packages: - numpy - "matplotlib==3.5.1"

The user needs to put all required python packages for the Bento Service in
a ``requirements.txt``. For a project, you can run ``pip freeze > requirements.txt``
to generate a requirements file to load with BentoML.

.. code:: yaml

python:
requirements_txt: "requirements.txt"

Additionally, there are more fields that can help manage larger projects.

.. code:: yaml

python:
requirements_txt: "requirements.txt"
lock_packages: False
index_url: "https://example.org/"
no_index: False
trusted_host: "localhost"
find_links: - "https://test.org/"
extra_index_url: - "https://test.org/"
pip_args: "--quiet"
wheels: - "./libs/my_package.whl"

By default, when the BentoML service generates package requirements
from the Bentofile, the package versions will be locked for easier
reproducibility.

If the `requirements.txt` includes locked packages, or a configuration
you need, set the `lock_packages` field to False.

If you're maintaining a private pip wheel, it can be included
with the `wheels` field.

If the wheel is hosted on a local network without TLS, you can indicate
that the domain is safe to pip with the `trusted_host` field.

## Conda Options

Similarly to PyPi, you can use Conda to handle dependencies.

.. code:: yaml

conda:
dependencies: - "scikit-learn==1.2.0" - numpy - nltk
channels: - "conda-forge"

Here, we need the conda-forge repository to install numpy with conda.
The `channels` field let's us specify that to the BentoML service.

In a preexisting environment, running `conda export` will generate
an `environment.yml` file to be included in the `environment_yml`
field.

.. code:: yaml

conda:
environment_yml: "environment.yml"

## Docker Options

BentoML makes it easy to deploy a Bento to a Docker container.

Here's a basic Docker options configuration.

.. code:: yaml

docker:
distro: debian
gpu: True
python_version: "3.8.9"
setup_script: "setup.sh"

For the `distro` options, you can choose from 5.

- debian
- amazonlinux2
- alpine
- ubi8
- ubi7

This config can be explored from `BentoML's Docker page <https://hub.docker.com/r/bentoml/bento-server>`\_.

The `gpu` field will also allocate a GPU in the Docker.
If you're using the standard devices variable in PyTorch,
for example, this field will enable the gpu.

For more interesting docker development, you can also use a
`setup.sh` for the container. For NLP projects, you can
preinstall NLTK data you need with:

.. code:: shell
   # ``setup.sh``
   python -m nltk.downloader all

Anatomy of a Bentofile
----------------------

+-------------+----------+---------------+
| Field       | Subfield | Default Value |
+-------------+----------+---------------+
| service     |          |               |
+-------------+----------+---------------+
| description |          |               |
+-------------+----------+---------------+
| labels      |          |               |
+-------------+----------+---------------+

Conclusion
----------

The `bentofile.yaml` is essential when generating a Bento,
and can be as simple or in-depth as you need. All configuration
can be included in the single file, or split with other smaller
requirements files.
