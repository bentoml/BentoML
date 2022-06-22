=====================
Dockerfile generation
=====================

This advanced guide describes the Dockerfile generation process. BentoML make uses
of `Jinja2 <https://jinja.palletsprojects.com/en/3.1.x/>`_, which enable users
to have more control on the generated Dockerfile for a Bento.

.. warning::

    This section is not meant to be a complete reference on Jinja2. It is meant to
    give a quick overview of how Jinja2 is used in conjunction with BentoML. For
    any reference on Jinja2 please refers to their `Templates Design Documentation <https://jinja.palletsprojects.com/en/3.1.x/templates/>`_.

.. tip::

    One huge advantage of this feature is that it allows developers and library,
    such as `🚀 bentoctl <https://github.com/bentoml/bentoctl>`_ to extend a 🍱.

The Dockerfile generation process is as follows:
    - Retrieve options provided via :code:`bentofile.yaml` :code:`docker` section.
    - Determine a given :class:`DistroSpec` for the coresponding distros. This
      includes supported python version, cuda version, and architecture type.
    - From the given spec, alongside with options from :class:`BentoBuildConfig`,
      generate a Dockerfile from a set of Jinja templates.

One of the powerful features Jinja offers is its `template inheritance <https://jinja.palletsprojects.com/en/3.1.x/templates/#template-inheritance>`_.
This allows BentoML to enable users to fully customize how to structure a Bento's Dockerfile.

.. note::

    To use a custom Dockerfile template, users have to provide a file with a format
    that follows the Jinja2 template syntax. The template file should have
    extensions of :code:`.j2`, :code:`.template`, :code:`.jinja`.

.. tip::

   :bdg-warning:`Warning:` If you pass in a generic :code:`Dockerfile` file, and then run :code:`bentoml build` to build a Bento and it doesn't throw any errors.
   However, when you try to run :code:`bentoml containerize`, this won't work.

   This is an expected behaviour from Jinja2, where Jinja2 accepts **any file** as a template.

   We decided not to put any restrictions to validate the template file, simply because we want to enable 
   users to customize to their own needs. 

   So, make sure you know what you are doing if you are trying out something like above.

In addition to Jinja, BentoML also uses `Docker Buildx <https://docs.docker.com/desktop/multi-arch/>`_, which enables users to build Bentos that support
multiple architectures. Under the hood the generated Dockefile leverage
`buildkit <https://github.com/moby/buildkit>`_, which enables some advanced `dockerfile features <https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md#dockerfile-frontend-syntaxes>`_.

.. note::

   Currently the only buildkit frontend that we are supporting is
   :code:`docker/dockerfile:1.4-labs`. Please contact us if you need support for
   different build frontend.


Writing custom Dockerfile template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


To construct a custom :code:`Dockerfile` template, users have to provide an `extends block <https://jinja.palletsprojects.com/en/3.1.x/templates/#extends>`_ at the beginning of the Dockerfile template :code:`Dockerfile.template`, followed by the given base template name:

.. code-block:: jinja

    {% extends "python_debian.j2" %}


The following templates per distro are available for extending:

.. dropdown:: :code:`alpine`
   :class-title: sd-text-primary

   .. tab-set::

       .. tab-item:: base_alpine.j2

           .. literalinclude:: ../../../bentoml/_internal/bento/docker/templates/base_alpine.j2
               :language: jinja

       .. tab-item:: miniconda_alpine.j2

           .. literalinclude:: ../../../bentoml/_internal/bento/docker/templates/miniconda_alpine.j2
               :language: jinja

       .. tab-item:: python_alpine.j2

           .. literalinclude:: ../../../bentoml/_internal/bento/docker/templates/python_alpine.j2
               :language: jinja

.. dropdown:: :code:`debian`
   :class-title: sd-text-primary

   .. tab-set::

       .. tab-item:: base_debian.j2

           .. literalinclude:: ../../../bentoml/_internal/bento/docker/templates/base_debian.j2
               :language: jinja

       .. tab-item:: miniconda_debian.j2

           .. literalinclude:: ../../../bentoml/_internal/bento/docker/templates/miniconda_debian.j2
               :language: jinja


       .. tab-item:: cuda_debian.j2

           .. literalinclude:: ../../../bentoml/_internal/bento/docker/templates/cuda_debian.j2
               :language: jinja

           .. note::

              For :code:`cuda_debian.j2`, we are using NVIDIA's `nvidia/cuda` image with
              ubuntu variants. This is because NVIDIA does not maintain a debian image.
              Ubuntu is a good substitute for Debian as Ubuntu is debian-based.

       .. tab-item:: python_debian.j2

           .. literalinclude:: ../../../bentoml/_internal/bento/docker/templates/python_debian.j2
               :language: jinja


.. dropdown:: Adding `conda` to CUDA-based template
    :class-title: sd-text-primary

    .. tip::

       :bdg-warning:`Warning:` miniconda install scripts provided by ContinuumIO (the parent company of Anaconda) supports Python 3.7 to 3.9. Make sure that you are using the correct python version under :code:`docker.python_version`.

    If you need to use conda for CUDA images, use the following template (*partially extracted from* `ContinuumIO/docker-images <https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile>`_):

    .. dropdown:: Expands me
       :class-title: sd-text-primary

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


.. dropdown:: About BentoML's templates inheritance:
   :icon: bookmark

   All internal templates are located `here <https://github.com/bentoml/BentoML/tree/main/bentoml/_internal/bento/docker/templates>`_

   As you can see, the BentoML internal Dockerfile templates are organized with the format :code:`<release_type>_<distro>.j2` with:

   +---------------+------------------------------------------+
   | Release type  | Description                              |
   +===============+==========================================+
   | base          | A base setup for all supported distros.  |
   +---------------+------------------------------------------+
   | cuda          | CUDA-supported templates.                |
   +---------------+------------------------------------------+
   | miniconda     | Conda-supported templates.               |
   +---------------+------------------------------------------+
   | python        | Python releases.                         |
   +---------------+------------------------------------------+

   where :code:`base_<distro>.j2` is extended from `base.j2 <https://github.com/bentoml/BentoML/tree/main/bentoml/_internal/bento/docker/templates/base.j2>`_

   The templates hierarchy is as follows:

   .. code-block:: bash

       .
       └── base.j2
           ├── base_alpine.j2
           │   ├── miniconda_alpine.j2
           │   └── python_alpine.j2
           ├── base_amazonlinux.j2
           │   └── python_amazonlinux.j2
           ├── base_debian.j2
           │   ├── cuda_debian.j2
           │   ├── miniconda_debian.j2
           │   └── python_debian.j2
           └── base_ubi8.j2
               ├── cuda_ubi8.j2
               └── python_ubi8.j2


Adding `conda` to CUDA-based template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tip::

   :bdg-warning:`Warning:` miniconda install scripts provided by ContinuumIO (the parent company of Anaconda) supports Python 3.7 to 3.9. Make sure that you are using the correct python version under :code:`docker.python_version`.

If you need to use conda for CUDA images, use the following template (*partially extracted from* `ContinuumIO/docker-images <https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile>`_):

.. dropdown:: Expands
   :icon: code
   :class-title: sd-text-primary

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


About BentoML's templates inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All internal templates are located `here <https://github.com/bentoml/BentoML/tree/main/bentoml/_internal/bento/docker/templates>`_.

As you can see, the BentoML internal Dockerfile templates are organized with the format :code:`<release_type>_<distro>.j2` with:

+---------------+------------------------------------------+
| Release type  | Description                              |
+===============+==========================================+
| base          | A base setup for all supported distros.  |
+---------------+------------------------------------------+
| cuda          | CUDA-supported templates.                |
+---------------+------------------------------------------+
| miniconda     | Conda-supported templates.               |
+---------------+------------------------------------------+
| python        | Python releases.                         |
+---------------+------------------------------------------+

where :code:`base_<distro>.j2` is extended from `base.j2 <https://github.com/bentoml/BentoML/tree/main/bentoml/_internal/bento/docker/templates/base.j2>`_

.. tip::

    BentoML also provides a Jinja2 global object called ``bento_base_template``
    which will determine automatically which templates should be used based on
    user docker options under :obj:`bentofile.yaml`:

    .. code-block:: jinja

        {% extends bento_base_template %}

By either using ``bento_base_template`` or extending any of the given base
templates, the generated Dockerfile will ensure to run any bento
correspondingly.

.. dropdown:: About writing :code:`Dockerfile.template`
   :icon: code
   :open:

   The Dockerfile template is a mix between :code:`Jinja2` syntax and :code:`Dockerfile`
   syntax. BentoML set both `trim_blocks` and `lstrip_blocks` in Jinja
   templates environment to :code:`True`. 

   Make sure that your Dockerfile instruction is **unindented** as if you are writting a normal Dockerfile.

   Refers to `Jinja Whitespace Control <https://jinja.palletsprojects.com/en/3.1.x/templates/#whitespace-control>`_.

   An example of a Dockerfile template takes advantage of multi-stage build to
   isolate the installation of a local library :code:`mypackage`:

   .. code-block:: dockerfile

      {% extends bento_autotemplate %}
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

Blocks
^^^^^^

BentoML defines a sets of `Blocks <https://jinja.palletsprojects.com/en/3.1.x/templates/#base-template>`_ under the object :code:`bento_autotemplate`.

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
any point inside block. This will ensure that the block is inherited from the
main :code:`SETUP_BENTO` block defined by BentoML.

The following are examples of how to use custom blocks:

.. tab-set::

    .. tab-item:: Example 1

        .. code:: jinja

           {% extends bento_autotemplate %}
           {% block SETUP_BENTO_USER %}
           {{ super() }}
           ENV CUSTOM_USER_VAR=foobar
           {% endblock %}

    .. tab-item:: Example 2

        .. code:: jinja

           {% extends bento_autotemplate %}
           {% block SETUP_BENTO_COMPONENTS %}
           RUN --mount=type=ssh git clone git@github.com:myorg/myproject.git myproject
           {{ super() }}
           {% endblock %}

    .. tab-item:: Example 3

        .. code:: jinja

           {% extends bento_autotemplate %}
           {% block SETUP_BENTO_BASE_IMAGE %}
           FROM --platform=$BUILDPLATFORM tensorflow/tensorflow:latest-devel as tf
           {{ super() }}
           COPY --from=tf /tf /tf
           ...
           {% endblock %}

An example of a custom Dockerfile template:

.. code-block:: jinja

    {% extends bento_autotemplate %}
    {% set bento__home = "/tmp" %}
    {% block SETUP_BENTO_ENTRYPOINT %}
    {{ super() }}
    RUN --mount=type=cache,mode=0777,target=/root/.cache/pip \
        pip install awslambdaric==2.0.0 mangum==0.12.3

    ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]

    {% endblock %}


.. dropdown:: About setting up 🍱 :code:`ENTRYPOINT` and :code:`CMD`
   :icon: code
   :color: light

   As you seen from the example above, the flexibility of a Jinja template also
   brings up the flexibility of setting up :code:`ENTRYPOINT` and :code:`CMD`.

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

        {% extends bento_autotemplate %}
        {% block SETUP_BENTO_ENTRYPOINT %}
        {{ super() }}

        ...
        ENTRYPOINT [ "{{ bento__entrypoint }}", "python", "-m", "awslambdaric" ]
        {% endblock %}

   .. tip::

        :code:`{{ bento__entrypoint }}` is the path the BentoML entrypoint,
        nothinig special here 😏

   Read more about :code:`CMD` and :code:`ENTRYPOINT` interaction `here <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`_.

Customizing Bento variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the example docker file above, we can see that we are also setting :code:`bento__home` variable.
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


.. admonition:: Help us improve the project!

    Found an issue or a TODO item? You're always welcome to make contributions to the
    project and its documentation. Check out the
    `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.
