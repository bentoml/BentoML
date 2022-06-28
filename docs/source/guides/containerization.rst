================
Containerization
================

This guide describes the containerization process and how users can
define a custom Dockerfile templates to customize Bento container. 


.. note::

    BentoML make uses of `Jinja2 <https://jinja.palletsprojects.com/en/3.1.x/>`_ to enable custom Dockerfile template.

    However, This section is not meant to be a complete reference on Jinja2.
    It is meant to give a quick overview of how Jinja2 is used in conjunction with BentoML.
    For any reference on Jinja2 please refers to their `Templates Design Documentation <https://jinja.palletsprojects.com/en/3.1.x/templates/>`_.

Why do you need this?
---------------------

1. If you want to customize the containerization process of your Bento.
2. If you need a certain tools, configs, prebuilt binaries that is available across all your Bento generated container images.
3. A big difference with :ref:`base image <concepts/bento:Docker Options Table>` features is that you don't have to setup a custom base image and then push it to a remote registry.

How it works
------------

To focus on how to create a custom Dockerfile template, the following examples
are provided:

1. :ref:`guides/containerization:Building binary into Bentos`
2. :ref:`guides/containerization:Access AWS credentials during image build`
3. :ref:`guides/containerization:Installing custom CUDA version with conda`

Building binary into Bentos
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's start with an example that builds an `Ingress <https://github.com/kubernetes/ingress-nginx>`_ binary into a Bento.

Define the following :code:`Dockerfile.template`:

.. code-block:: Dockerfile

   {% extends bento_base_template %}
   {% block SETUP_BENTO_BASE_IMAGE %}

   ARG ARCH=amd64

   FROM golang:alpine as build-stage

   ARG ARCH

   WORKDIR /tmp

   RUN --mount=type=cache,target=/var/cache/apk \
       apk --update add bash build-base git

   SHELL [ "/bin/bash", "-exo", "pipefail", "-c" ]

   RUN git clone --depth 1 https://github.com/kubernetes/ingress-nginx.git

   WORKDIR /tmp/ingress-nginx

   RUN --mount=type=cache,target=/root/.cache/go-build \
       PKG=k8s.io/ingress-nginx \
       ARCH=${ARCH} \
       COMMIT_SHA=$(git rev-parse --short HEAD) \
       REPO_INFO=$(git config --get remote.origin.url) \
       TAG="0.0.0" \
       ./build/build.sh

   WORKDIR /tmp/ingress-nginx/rootfs/bin/${ARCH}

   {{ super() }}

   ARG ARCH

   COPY --from=build-stage /tmp/ingress-nginx/rootfs/bin/${ARCH}/ /usr/local/bin

   {% endblock %}

Then add the following to your :code:`bentofile.yaml`:

.. code-block:: yaml

   docker:
     system_packages:
       - nginx
     dockerfile_template: ./Dockerfile.template

Proceed to build your Bento with :code:`bentoml build` and containerize with :code:`bentoml containerize`:

.. code-block:: bash

   bentoml build

   bentoml containerize <bento>:<tag>

.. tip:: 

   You can also provide :code:`--progress plain` to see the progress from
   `buildkit <https://github.com/moby/buildkit>`_ in plain text

   .. code-block:: yaml

      bentoml containerize --progress plain <bento>:<tag>`

Access AWS credentials during image build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will now demonstrate how to provide AWS credentials to a Bento via two approaches:

1. :ref:`guides/containerization:Using :code:`ARG``.
2. :ref:`guides/containerization:Using Docker's :code:`RUN --mount``.

.. note::

   :bdg-info:`Remarks:` We recommend for most cases 
   to use the second option (:ref:`guides/containerization:Using Docker's :code:`RUN --mount``)
   as it prevents any securities leak.

   By default BentoML uses the latest `dockerfile frontend <https://hub.docker.com/r/docker/dockerfile>`_ which
   allows mounting secrets to container.

For both examples, you will need to add the following to your :code:`bentofile.yaml`:

.. code-block:: yaml

   python:
     packages:
     - awscli
   docker:
     dockerfile_template: ./Dockerfile.template


Using :code:`ARG`
^^^^^^^^^^^^^^^^^

.. note::

   Courtesy of the works from `Mission Lane <https://www.missionlane.com/>`_ Engineering Team.

Define the following :code:`Dockerfile.template`:

.. code-block:: Dockerfile

   {% extends bento_base_template %}
   {% block SETUP_BENTO_BASE_IMAGE %}
   ARG AWS_SECRET_ACCESS_KEY
   ARG AWS_ACCESS_KEY_ID
   {{ super() }}

   ARG AWS_SECRET_ACCESS_KEY
   ARG AWS_ACCESS_KEY_ID
   {% endblock %}
   {% block SETUP_BENTO_COMPONENTS %}
   {{ super() }}

   RUN aws s3 cp s3://path/to/file {{ bento__path }}

   {% endblock %}

After building the bento with :code:`bentoml build`, you can then
pass :code:`AWS_SECRET_ACCESS_KEY` and :code:`AWS_ACCESS_KEY_ID` as arguments to :code:`bentoml containerize`:

.. code-block:: bash

   bentoml containerize --build-arg AWS_SECRET_ACCESS_KEY=<secret_access_key> --build-arg AWS_ACCESS_KEY_ID=<access_key_id> <bento>:<tag>

.. note::

   We recommend not to do this practice as anyone whos has access to the
   history of the computer running the aboved command can access the AWS credentials. Instead
   use :ref:`guides/containerization:Using Docker's :code:`RUN --mount``.

Using Docker's :code:`RUN --mount`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the following :code:`Dockerfile.template`:

.. code-block:: Dockerfile

   {% extends bento_base_template %}
   {% block SETUP_BENTO_COMPONENTS %}
   {{ super() }}
   
   RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
        aws s3 cp s3://path/to/file {{ bento__path }}

   {% endblock %}

Follow the above addition to :code:`bentofile.yaml` to include ``awscli`` and
the custom dockerfile template.

To pass in secrets to the Bento, pass it via :code:`--secret` to :code:`bentoml
containerize`:

.. code-block:: bash

   bentoml containerize --secret id=aws,src=$HOME/.aws/credentials <bento>:<tag>

.. seealso::

   `Mounting Secrets <https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md#run---mounttypesecret>`_

Installing custom CUDA version with conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lastly, we will also demonstrate how you can install custom cuda version via
conda.

Add the following to your :code:`bentofile.yaml`:

.. code-block:: yaml

   conda:
     channels:
     - conda-forge
     - nvidia
     - defaults
     dependencies:
     - cudatoolkit-dev=10.1
     - cudnn=7.6.4
     - cxx-compiler=1.0
     - mpi4py=3.0 # installs cuda-aware openmpi
     - matplotlib=3.2
     - networkx=2.4
     - numba=0.48
     - pandas=1.0

Then proceed with :code:`bentoml build` and :code:`bentoml containerize`
respectively:

.. code-block:: bash

   bentoml build

   bentoml containerize <bento>:<tag>

Writing :code:`dockerfile_template`
-----------------------------------

BentoML utilize `Jinja2 <https://jinja.palletsprojects.com/en/3.1.x/>`_ to
structure a :code:`Dockerfile.template`.

The Dockerfile template is a mix between :code:`Jinja2` syntax and :code:`Dockerfile`
syntax. BentoML set both `trim_blocks` and `lstrip_blocks` in Jinja
templates environment to :code:`True`. 

.. note::

   Make sure that your Dockerfile instruction is **unindented** as if you are writting a normal Dockerfile.

.. seealso::

    `Jinja Whitespace Control <https://jinja.palletsprojects.com/en/3.1.x/templates/#whitespace-control>`_.


An example of a Dockerfile template takes advantage of multi-stage build to
isolate the installation of a local library :code:`mypackage`:

.. code-block:: dockerfile

   {% extends bento_base_template %}
   {% block SETUP_BENTO_BASE_IMAGE %}
   FROM --platform=$BUILDPLATFORM python:3.7-slim as buildstage
   RUN mkdir /tmp/mypackage

   WORKDIR /tmp/mypackage/
   COPY mypackage .
   RUN python setup.py sdist && mv dist/mypackage-0.0.1.tar.gz mypackage.tar.gz

   {{ super() }}
   {% endblock %}
   {% block SETUP_BENTO_COMPONENTS %}
   {{ super() }}
   COPY --from=buildstage mypackage.tar.gz /tmp/wheels/
   RUN --network=none pip install --find-links /tmp/wheels mypackage
   {% endblock %}

.. note::
   Notice how for all Dockerfile instruction, we consider as if the Jinja
   logics aren't there 🚀.


Jinja templates
~~~~~~~~~~~~~~~

One of the powerful features Jinja offers is its `template inheritance <https://jinja.palletsprojects.com/en/3.1.x/templates/#template-inheritance>`_.
This allows BentoML to enable users to fully customize how to structure a Bento's Dockerfile.

.. note::

   To use a custom Dockerfile template, users have to provide a file with a format
   that follows the Jinja2 template syntax. The template file should have
   extensions of :code:`.j2`, :code:`.template`, :code:`.jinja`.

To construct a custom :code:`Dockerfile` template, users have to provide an `extends block <https://jinja.palletsprojects.com/en/3.1.x/templates/#extends>`_ at the beginning of the Dockerfile template :code:`Dockerfile.template` followed by the given base template name :code:`bento_base_template`:

.. code-block:: jinja

   {% extends bento_base_template %}

.. tip::

   :bdg-warning:`Warning:` If you pass in a generic :code:`Dockerfile` file, and then run :code:`bentoml build` to build a Bento and it doesn't throw any errors.

   However, when you try to run :code:`bentoml containerize`, this won't work.

   This is an expected behaviour from Jinja2, where Jinja2 accepts **any file** as a template.

   We decided not to put any restrictions to validate the template file, simply because we want to enable 
   users to customize to their own needs. 

:code:`{{ super() }}`
^^^^^^^^^^^^^^^^^^^^^

As you can notice throughout this guides, we use a special function :code:`{{ super() }}`. This is a Jinja
features that allow users to call content of `parent block <https://jinja.palletsprojects.com/en/3.1.x/templates/#super-blocks>`_. This 
enables users to fully extend base templates provided by BentoML to ensure that
the result Bentos can be containerized.

.. seealso::

   |super_tag|_ for more information on template inheritance.

.. _super_tag: https://jinja.palletsprojects.com/en/3.1.x/templates/#super-blocks

.. |super_tag| replace:: :code:`{{ super() }}` *Syntax*

Blocks
^^^^^^

BentoML defines a sets of `Blocks <https://jinja.palletsprojects.com/en/3.1.x/templates/#base-template>`_ under the object :code:`bento_base_template`.

All exported blocks that users can use to extend are as follow:

+---------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| Blocks                          | Definition                                                                                                                       |
+=================================+==================================================================================================================================+
| :code:`SETUP_BENTO_BASE_IMAGE`  | Instructions to set up multi architecture supports, base images as well as installing system packages that is defined by users.  |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| :code:`SETUP_BENTO_USER`        | Setup bento users with correct UID, GID and directory for a 🍱.                                                                  |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| :code:`SETUP_BENTO_ENVARS`      | Add users environment variables (if specified) and other required variables from BentoML.                                        |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| :code:`SETUP_BENTO_COMPONENTS`  | Setup components for a 🍱 , including installing pip packages, running setup scripts, installing bentoml, etc.                   |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| :code:`SETUP_BENTO_ENTRYPOINT`  | Finalize ports and set :code:`ENTRYPOINT` and :code:`CMD` for the 🍱.                                                            |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------------+

.. note::

   All the defined blocks are prefixed with :code:`SETUP_BENTO_*`. This is to
   ensure that users can extend blocks defined by BentoML without sacrificing
   the flexibility of a Jinja template.

To extend any given block, users can do so by adding :code:`{{ super() }}` at
any point inside block.


Dockerfile instruction
~~~~~~~~~~~~~~~~~~~~~~

.. seealso::

   `Dockerfile reference <https://docs.docker.com/engine/reference/builder>`_ for writing a Dockerfile.

We recommend that users should use the following Dockerfile instructions in
their custom Dockerfile templates: :code:`ENV`, :code:`RUN`, :code:`ARG`. These
instructions are mostly used and often times will get the jobs done.

There are a few instructions that you shouldn't use unless you know what you are
doing:

+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Instruction    | Reasons not to use                                                                                                                                                                                                                                        |
+================+===========================================================================================================================================================================================================================================================+
| :code:`FROM`   | Since the containerized Bento is a multi-stage builds container, adding :code:`FROM` statement will result in failure to containerize the given Bento.                                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`SHELL`  | BentoML uses `heredoc syntax <https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md#user-content-here-documents>`_ and using :code:`bash` in our containerization process. Hence changing :code:`SHELL` will result in failure. |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`CMD`    | Changing :code:`CMD` will inherently modify the behaviour of the bento container where docker won't be able to run the bento inside the container. More below                                                                                             |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

The following instructions should be used with caution:


:code:`WORKDIR`
^^^^^^^^^^^^^^^

.. seealso::

   `WORKDIR reference <https://docs.docker.com/engine/reference/builder/#workdir>`_

Since :code:`WORKDIR` determines the working directory for any :code:`RUN`, :code:`CMD`, :code:`ENTRYPOINT`, :code:`COPY` and :code:`ADD` instructions that follow it in the Dockerfile,
make sure that your instructions define the correct path to any working files.

.. note::

   By default, all paths for Bento-related files will be generated to its
   fspath, which ensures that Bento will work regardless of :code:`WORKDIR`


:code:`ENTRYPOINT`
^^^^^^^^^^^^^^^^^^

.. seealso::

   `ENTRYPOINT reference <https://docs.docker.com/engine/reference/builder/#entrypoint>`_


The flexibility of a Jinja template also brings up the flexibility of setting up :code:`ENTRYPOINT` and :code:`CMD`.

From `Dockerfile documentation <https://docs.docker.com/engine/reference/builder/#entrypoint>`_:

    Only the last :code:`ENTRYPOINT` instruction in the Dockerfile will have an effect.

By default, a Bento sets:

.. code-block:: dockerfile

    ENTRYPOINT [ "{{ bento__entrypoint }}" ]

    CMD ["bentoml", "serve", "{{ bento__path }}", "--production"]

This means that if you have multiple :code:`ENTRYPOINT` instructions, you will have to
make sure the last :code:`ENTRYPOINT` will run bentoml when using :code:`docker
run` on the 🍱 container. 

In cases where one needs to setup different :code:`ENTRYPOINT`, you can use
the :code:`ENTRYPOINT` instruction under the :code:`SETUP_BENTO_ENTRYPOINT` block as follow:

.. code-block:: jinja

    {% extends bento_base_template %}
    {% block SETUP_BENTO_ENTRYPOINT %}
    {{ super() }}

    ...
    ENTRYPOINT [ "{{ bento__entrypoint }}", "python", "-m", "awslambdaric" ]
    {% endblock %}

.. tip::

    :code:`{{ bento__entrypoint }}` is the path the BentoML entrypoint,
    nothinig special here 😏.

Read more about :code:`CMD` and :code:`ENTRYPOINT` interaction `here <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`_.

Advanced Options
----------------

The next part goes into advanced options. Skip this part if you are not
comfortable with using it.

Dockerfile variables
~~~~~~~~~~~~~~~~~~~~

BentoML does expose some variables that user can modify to fit their needs.

The following are the variables that users can set in their custom Dockerfile template:

+-------------------------+---------------------------------------------------------------------+
| Variables               | Description                                                         |
+=========================+=====================================================================+
| :code:`bento__home`     | Setup bento home, default to :code:`/home/{{ bento__user }}`        |
+-------------------------+---------------------------------------------------------------------+
| :code:`bento__user`     | Setup bento user, default to :code:`bentoml`                        |
+-------------------------+---------------------------------------------------------------------+
| :code:`bento__uid_gid`  | Setup UID and GID for the user, default to :code:`1034:1034`        |
+-------------------------+---------------------------------------------------------------------+
| :code:`bento__path`     | Setup bento path, default to :code:`/home/{{ bento__user }}/bento`  |
+-------------------------+---------------------------------------------------------------------+

If any of the aforementioned fields are set with :code:`{% set ... %}`, then we
will use your value instead, otherwise a default value will be used.

Adding :code:`conda` to CUDA-enabled Bento
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tip::

   :bdg-warning:`Warning:` miniconda install scripts provided by ContinuumIO (the parent company of Anaconda) supports Python 3.7 to 3.9. Make sure that you are using the correct python version under :code:`docker.python_version`.

If you need to use conda for CUDA images, use the following template ( *partially extracted from* `ContinuumIO/docker-images <https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile>`_ ):

.. dropdown:: Expands me
   :class-title: sd-text-primary
   :icon: code

   .. code-block:: jinja

      {% extends "base_debian.j2" %}
      {# Make sure to change the correct python_version and conda version accordingly. #}
      {# example: py38_4.10.3 #}
      {# refers to https://repo.anaconda.com/miniconda/ for miniconda3 base #}
      {% set conda_version="py39_4.11.0" %}
      {% set conda_path="/opt/conda" %}
      {% set conda_exec= [conda_path, "bin", "conda"] | join("/") %}
      {% block SETUP_BENTO_BASE_IMAGE %}
      FROM debian:bullseye-slim as conda-build

      RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/apt \
          --mount=type=cache,from=cached,sharing=shared,target=/var/lib/apt \
          apt-get update -y && \
          apt-get install -y --no-install-recommends --allow-remove-essential \
                      software-properties-common \
              bzip2 \
              ca-certificates \
              git \
              libglib2.0-0 \
              libsm6 \
              libxext6 \
              libxrender1 \
              mercurial \
              openssh-client \
              procps \
              subversion \
              wget && \
          apt-get clean

      ENV PATH {{ conda_path }}/bin:$PATH

      SHELL [ "/bin/bash", "-eo", "pipefail", "-c" ]

      ARG CONDA_VERSION={{ conda_version }}

      RUN bash <<EOF
      set -ex

      UNAME_M=$(uname -m)

      if [ "${UNAME_M}" = "x86_64" ]; then
          MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh";
          SHA256SUM="4ee9c3aa53329cd7a63b49877c0babb49b19b7e5af29807b793a76bdb1d362b4";
      elif [ "${UNAME_M}" = "s390x" ]; then
          MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh";
          SHA256SUM="e5e5e89cdcef9332fe632cd25d318cf71f681eef029a24495c713b18e66a8018";
      elif [ "${UNAME_M}" = "aarch64" ]; then
          MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh";
          SHA256SUM="00c7127a8a8d3f4b9c2ab3391c661239d5b9a88eafe895fd0f3f2a8d9c0f4556";
      elif [ "${UNAME_M}" = "ppc64le" ]; then
          MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-ppc64le.sh";
          SHA256SUM="8ee1f8d17ef7c8cb08a85f7d858b1cb55866c06fcf7545b98c3b82e4d0277e66";
      fi

      wget "${MINICONDA_URL}" -O miniconda.sh -q && echo "${SHA256SUM} miniconda.sh" > shasum

      if [ "${CONDA_VERSION}" != "latest" ]; then 
          sha256sum --check --status shasum; 
      fi

      mkdir -p /opt
      sh miniconda.sh -b -p {{ conda_path }} && rm miniconda.sh shasum

      find {{ conda_path }}/ -follow -type f -name '*.a' -delete
      find {{ conda_path }}/ -follow -type f -name '*.js.map' -delete
      {{ conda_exec }} clean -afy
      EOF

      {{ super() }}

      ENV PATH {{ conda_path }}/bin:$PATH

      COPY --from=conda-build {{ conda_path }} {{ conda_path }}

      RUN bash <<EOF
      ln -s {{ conda_path }}/etc/profile.d/conda.sh /etc/profile.d/conda.sh
      echo ". {{ conda_path }}/etc/profile.d/conda.sh" >> ~/.bashrc
      echo "{{ conda_exec }} activate base" >> ~/.bashrc
      EOF

      {% endblock %}
      {% block SETUP_BENTO_ENVARS %}

      SHELL [ "/bin/bash", "-eo", "pipefail", "-c" ]

      {{ super() }}

      RUN --mount=type=cache,mode=0777,target=/opt/conda/pkgs bash <<EOF
      SAVED_PYTHON_VERSION={{ __python_version_full__ }}
      PYTHON_VERSION=${SAVED_PYTHON_VERSION%.*}

      echo "Installing Python $PYTHON_VERSION with conda..."
      {{ conda_exec }} install -y -n base pkgs/main::python=$PYTHON_VERSION pip

      if [ -f {{ __environment_yml__ }} ]; then
      # set pip_interop_enabled to improve conda-pip interoperability. Conda can use
      # pip-installed packages to satisfy dependencies.
      echo "Updating conda base environment with environment.yml"
      {{ conda_exec }} config --set pip_interop_enabled True || true
      {{ conda_exec }} env update -n base -f {{ __environment_yml__ }}
      {{ conda_exec }} clean --all
      fi
      EOF
      {% endblock %}
