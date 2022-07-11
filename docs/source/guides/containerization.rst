================
Containerization
================

This guides describes advanced containerization options 
provided by BentoML:

- :ref:`Using base image <guides/containerization:Custom Base Image>`
- :ref:`Using dockerfile template <guides/containerization:Dockerfile Template>`

This is an advanced feature for user to customize container environment that are not directly supported in BentoML.
For basic containerizing options, see :ref:`Docker Options <concepts/bento:Docker Options>`.

Why you may need this?
----------------------

- If you want to customize the containerization process of your Bento.
- If you need a certain tools, configs, prebuilt binaries that is available across all your Bento generated container images.
- A big difference with :ref:`base image <concepts/bento:Docker Options Table>` features is that you don't have to setup a custom base image and then push it to a remote registry.

Custom Base Image
-----------------

If none of the provided distros work for your use case, e.g. if your infrastructure
requires all docker images to be derived from the same base image with certain security
fixes and libraries, you can config BentoML to use your base image instead:

.. code:: yaml

    docker:
        base_image: "my_custom_image:latest"

When a :code:`base_image` is provided, **all other docker options will be ignored**,
(distro, cuda_version, system_packages, python_version). :code:`bentoml containerize`
will build a new image on top of the base_image with the following steps:

- setup env vars
- run the :code:`setup_script` if provided
- install the required Python packages
- copy over the Bento file
- setup the entrypoint command for serving.


.. note::

    :bdg-warning:`Warning:` user must ensure that the provided base image has desired
    Python version installed. If the base image you have doesn't have Python, you may
    install python via a :code:`setup_script`. The implementation of the script depends
    on the base image distro or the package manager available.

    .. code:: yaml

        docker:
            base_image: "my_custom_image:latest"
            setup_script: "./setup.sh"

.. warning::

    By default, BentoML supports multi-platform docker image build out-of-the-box.
    However, when a custom :code:`base_image` is provided, the generated Dockerfile can
    only be used for building linux/amd64 platform docker images.

    If you are running BentoML from an Apple M1 device or an ARM based computer, make
    sure to pass the :code:`--platform` parameter when containerizing a Bento. e.g.:

    .. code:: bash

        bentoml containerize iris_classifier:latest --platform=linux/amd64


Dockerfile Template
-------------------

The :code:`dockerfile_template` field gives the user full control over how the
:code:`Dockerfile` is generated for a Bento by extending the template used by
BentoML.

First, create a :code:`Dockerfile.template` file next to your :code:`bentofile.yaml`
build file. This file should follow the
`Jinja2 <https://jinja.palletsprojects.com/en/3.1.x/>`_ template language, and extend
BentoML's base template and blocks. The template should render a valid
`Dockerfile <https://docs.docker.com/engine/reference/builder/>`_. For example:

.. code-block:: jinja

   {% extends bento_base_template %}
   {% block SETUP_BENTO_COMPONENTS %}
   {{ super() }}
   RUN echo "We are running this during bentoml containerize!"
   {% endblock %}

Then add the path to your template file to the :code:`dockerfile_template` field in
your :code: `bentofile.yaml`:

.. code:: yaml

    docker:
        dockerfile_template: "./Dockerfile.template"

Now run :code:`bentoml build` to build a new Bento. It will contain a Dockerfile
generated with the custom template. To confirm the generated Dockerfile works as
expected, run :code:`bentoml containerize <bento>` to build a docker image with it.

.. dropdown:: View the generated Dockerfile content
    :icon: code

    During development and debugging, you may want to see the generated Dockerfile.
    Here's shortcut for that:

    .. code-block:: bash

        cat "$(bentoml get <bento>:<tag> -o path)/env/docker/Dockerfile"

Examples
--------

1. :ref:`guides/containerization:Building Tensorflow custom op`
2. :ref:`guides/containerization:Access AWS credentials during image build`

Building Tensorflow custom op
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's start with an example that builds a `custom Tensorflow op <https://www.tensorflow.org/guide/create_op>`_ binary into a Bento, which is based on |zero_out|_:

.. _zero_out: https://www.tensorflow.org/guide/create_op#define_the_op_interface

.. |zero_out| replace:: :code:`zero_out.cc` implementation details


Define the following :code:`Dockerfile.template`:

.. code-block:: jinja

   {% extends bento_base_template %}
   {% block SETUP_BENTO_BASE_IMAGE %}

   {{ super() }}

   WORKDIR /tmp

   SHELL [ "bash", "-exo", "pipefail", "-c" ]

   COPY ./src/tfops/zero_out.cc .

   RUN pip3 install tensorflow
   RUN bash <<EOF
   set -ex

   TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
   TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

   g++ --std=c++14 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include());') -D_GLIBCXX_USE_CXX11_ABI=0 -O2
   EOF

   {% endblock %}
   {% block SETUP_BENTO_COMPONENTS %}
   {{ super() }}
   RUN stat /usr/lib/zero_out.so
   {% endblock %}


Then add the following to your :code:`bentofile.yaml`:

.. code-block:: yaml

   include:
     - "zero_out.cc"
   python:
     packages:
     - tensorflow
   docker:
     dockerfile_template: ./Dockerfile.template

Proceed to build your Bento with :code:`bentoml build` and containerize with :code:`bentoml containerize`:

.. code-block:: bash

   bentoml build

   bentoml containerize <bento>:<tag>

.. tip:: 

   You can also provide :code:`--progress plain` to see the progress from
   `buildkit <https://github.com/moby/buildkit>`_ in plain text

   .. code-block:: yaml

      bentoml containerize --progress plain <bento>:<tag>

Access AWS credentials during image build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will now demonstrate how to provide AWS credentials to a Bento via two approaches:

1. :ref:`guides/containerization:Using environment variables`.
2. :ref:`guides/containerization:Mount credentials from host`.

.. note::

   :bdg-info:`Remarks:` We recommend for most cases 
   to use the second option (:ref:`guides/containerization:Mount credentials from host`)
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


Using environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the following :code:`Dockerfile.template`:

.. code-block:: jinja

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

   bentoml containerize --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
                        --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
                        <bento>:<tag>

Mount credentials from host
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the following :code:`Dockerfile.template`:

.. code-block:: jinja

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

.. code-block:: jinja

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

.. note::

   This section is not meant to be a complete reference on Jinja2.
   For any advanced features from on Jinja2, please refers to their `Templates Design Documentation <https://jinja.palletsprojects.com/en/3.1.x/templates/>`_.


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

The use of the following instructions can be **potentially harmful**. They should be reserved for specialized advanced use cases.

+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Instruction    | Reasons not to use                                                                                                                                                                                                                                        |
+================+===========================================================================================================================================================================================================================================================+
| :code:`FROM`   | Since the containerized Bento is a multi-stage builds container, adding :code:`FROM` statement will result in failure to containerize the given Bento.                                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`SHELL`  | BentoML uses `heredoc syntax <https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md#user-content-here-documents>`_ and using :code:`bash` in our containerization process. Hence changing :code:`SHELL` will result in failure. |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`CMD`    | Changing :code:`CMD` will inherently modify the behaviour of the bento container where docker won't be able to run the bento inside the container. More :ref:`below <guides/containerization:\:code\:\`entrypoint\`>`                                     |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

The following instructions should be **used with caution**:


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

.. code-block:: jinja

    ENTRYPOINT [ "{{ bento__entrypoint }}" ]

    CMD ["bentoml", "serve", "{{ bento__path }}", "--production"]

This aboved instructions ensure that whenever :code:`docker run` is invoked on the 🍱 container, :code:`bentoml` is called correctly. 

In scenarios where one needs to setup a custom :code:`ENTRYPOINT`, make sure to use
the :code:`ENTRYPOINT` instruction under the :code:`SETUP_BENTO_ENTRYPOINT` block as follows:

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

If you need to use conda for CUDA images, use the following template ( *partially extracted from* |conda_docker|_ ):

.. dropdown:: Expands me
   :class-title: sd-text-primary
   :icon: code

   .. code-block:: jinja

      {% import '_macros.j2' as common %}
      {% extends bento_base_template %}
      {# Make sure to change the correct python_version and conda version accordingly. #}
      {# example: py38_4.10.3 #}
      {# refers to https://repo.anaconda.com/miniconda/ for miniconda3 base #}
      {% set conda_version="py39_4.11.0" %}
      {% set conda_path="/opt/conda" %}
      {% set conda_exec=[conda_path, "bin", "conda"] | join("/") %}
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
      {{ common.setup_conda(__python_version__, bento__path, conda_path=conda_path) }}
      {% endblock %}

.. _conda_docker: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile

.. |conda_docker| replace:: :code:`ContinuumIO/docker-images`
