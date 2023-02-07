=========================
Advanced Containerization
=========================

*time expected: 12 minutes*

This guide describes advanced containerization options 
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
    sure to pass the :code:`--opt platform=linux/amd64` parameter when containerizing a Bento. e.g.:

    .. code:: bash

        bentoml containerize iris_classifier:latest --opt platform=linux/amd64


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

1. :ref:`guides/containerization:Building TensorFlow custom op`
2. :ref:`guides/containerization:Access AWS credentials during image build`

Building TensorFlow custom op
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's start with an example that builds a `custom TensorFlow op <https://www.tensorflow.org/guide/create_op>`_ binary into a Bento, which is based on |zero_out|_:

.. _zero_out: https://www.tensorflow.org/guide/create_op#define_the_op_interface

.. |zero_out| replace:: :code:`zero_out.cc` implementation details


Define the following :code:`Dockerfile.template`:

.. literalinclude:: ./snippets/containerization/tf_ops.template
   :language: jinja
   :caption: `Dockerfile.template`


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
   
   ENV AWS_SECRET_ACCESS_KEY=$ARG AWS_SECRET_ACCESS_KEY
   ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
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

   .. literalinclude:: ./snippets/containerization/conda_cuda.template
      :language: jinja
      :caption: `Dockerfile.template`

Containerization with different container engines.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In BentoML version 1.0.11 [#pr_3164]_, we support different container engines aside from docker.

BentoML-generated Dockerfiles from version 1.0.11 onward will be OCI-compliant and can be built with:

* `Docker <https://www.docker.com/>`_
* `Podman <https://podman.io/>`_
* `Buildah <https://buildah.io/>`_
* `nerdctl <https://github.com/containerd/nerdctl>`_
* :github:`buildctl <moby/buildkit/blob/master/docs/buildctl.md>`
* `Docker buildx <https://docs.docker.com/engine/reference/commandline/buildx/>`_

To use any of the aforementioned backends, they must be installed on your system. Refer to their documentation for installation and setup.

.. note::

   By default, BentoML will use Docker as the container backend. 
   To use other container engines, please set the environment variable ``BENTOML_CONTAINERIZE_BACKEND`` or
   pass in ``--backend`` to :ref:`bentoml containerize <reference/cli:containerize>`:

   .. code-block:: bash

      # set environment variable
      BENTOML_CONTAINERIZE_BACKEND=buildah bentoml containerize pytorch-mnist

      # or pass in --backend
      bentoml containerize pytorch-mnist:latest --backend buildah

To build a BentoContainer in Python, you can use the :ref:`Container SDK <reference/container:Container APIs>` method :meth:`bentoml.container.build`:

.. code-block:: python

   import bentoml

   bentoml.container.build(
      "pytorch-mnist:latest",
      backend="podman",
      features=["grpc","grpc-reflection"],
      cache_from="registry.com/my_cache:v1",
   )


Register custom backend
^^^^^^^^^^^^^^^^^^^^^^^

To register a new backend, there are two functions that need to be implemented:

* ``arg_parser_func``: a function that takes in keyword arguments that represents the builder
  commandline arguments and returns a ``list[str]``:

  .. code-block:: python

     def arg_parser_func(
         *,
         context_path: str = ".",
         cache_from: Optional[str] = None,
         **kwargs,
     ) -> list[str]:
         if cache_from:
             args.extend(["--cache-from", cache_from])
         args.append(context_path)
         return args

* ``health_func``: a function that returns a ``bool`` to indicate if the backend is available:

  .. code-block:: python

     import shutil

     def health_func() -> bool:
         return shutil.which("limactl") is not None

To register a new backend, use :meth:`bentoml.container.register_backend`:

.. code-block:: python

   from bentoml.container import register_backend

   register_backend(
      "lima",
      binary="/usr/bin/limactl",
      buildkit_support=True,
      health=health_func,
      construct_build_args=arg_parser_func,
      env={"DOCKER_BUILDKIT": "1"},
   )

.. dropdown:: Backward compatibility with ``bentoml.bentos.containerize``
   :class-title: sd-text-primary

   Before 1.0.11, BentoML uses :meth:`bentoml.bentos.containerize` to containerize Bento. This method is now deprecated and will be removed in the future.

BuildKit interop
^^^^^^^^^^^^^^^^

BentoML leverages `BuildKit <https://github.com/moby/buildkit>`_ for a more extensive feature set. However, we recognise that  
BuildKit has come with a lot of friction for migration purposes as well as restrictions to use with other build tools (such as podman, buildah, kaniko).

Therefore, since BentoML version 1.0.11, BuildKit will be an opt-out. To disable BuildKit, pass ``DOCKER_BUILDKIT=0`` to
:ref:`bentoml containerize <reference/cli:containerize>`, which aligns with the behaviour of ``docker build``:

.. code-block:: bash

    $ DOCKER_BUILDKIT=0 bentoml containerize ...

.. note::

    All Bento container will now be following OCI spec instead of Docker spec. The difference is that in OCI spec, there is no SHELL argument.

.. note::

   The generated Dockerfile included inside the Bento will be a minimal Dockerfile, which ensures compatibility among build tools. We encourage users to always use
   :ref:`bentoml containerize <reference/cli:containerize>`.

   *If you wish to use the generated Dockerfile, make sure that you know what you are doing!*

CLI enhancement
^^^^^^^^^^^^^^^

To better support different backends, :ref:`bentoml containerize <reference/cli:containerize>`
will be more agnostic when it comes to parsing options.

One can pass in options for specific backend with ``--opt``:

.. code-block:: bash

   $ bentoml containerize pytorch-mnist:latest --backend buildx --opt platform=linux/arm64

``--opt`` also accepts parsing ``:``

.. code-block:: bash

   $ bentoml containerize pytorch-mnist:latest --backend buildx --opt platform:linux/arm64

.. note::

   If you are seeing a warning message like:

   .. code-block:: prolog

       '--platform=linux/arm64' is now deprecated, use the equivalent '--opt platform=linux/arm64' instead.

   BentoML used to depends on Docker buildx. These options are now backward compatible with ``--opt``. You can safely ignore this warning and use
   ``--opt`` to pass options for ``--backend=buildx``.

----

.. rubric:: Notes

.. [#pr_3164] Introduction of container builder to build Bento into OCI-compliant image: :github:`bentoml/BentoML/pull/3164`

.. _conda_docker: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile

.. |conda_docker| replace:: :code:`ContinuumIO/docker-images`
