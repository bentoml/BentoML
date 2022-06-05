============
Build Config
============

Build Config are used to represent the structure of
a :code:`bentofile.yaml` in our Python API. 

.. warning::

    This page contains BentoML internal API reference. This documentation is
    provided for developers who wants to explore BentoML internals as well as
    extends BentoML features. 

.. note::

    Refers to :ref:`concepts/bento:Building Bentos` for more information about how to build a bento.

BentoML under the hood utilize `attrs library <https://www.attrs.org/en/stable/index.html>`_ 
to provide a better experience writing classes in Python. They comes with features, such as
validation, conversion, supports for type-checking, etc. 

All classes below are :code:`attrs.frozen()` class, meaning values are immutable to change.
Refers to `attrs.frozen() API reference <https://www.attrs.org/en/stable/api.html#attr.attrs.frozen>`_.


BentoBuildConfig
----------------

.. autoclass:: bentoml._internal.bento.build_config.BentoBuildConfig
    :members:


PythonOptions
-------------

.. autoclass:: bentoml._internal.bento.build_config.PythonOptions
    :members:


CondaOptions
------------

.. autoclass:: bentoml._internal.bento.build_config.CondaOptions
    :members:


DockerOptions
-------------

.. autoclass:: bentoml._internal.bento.build_config.DockerOptions
    :members:

Dockerfile generation
~~~~~~~~~~~~~~~~~~~~~

This next section describes the Dockerfile generation process. BentoML make uses
of `Jinja2 <https://jinja.palletsprojects.com/en/3.1.x/>`_, which enable users
to have more control on the generated Dockerfile for a Bento.

.. dropdown:: About this section
   :animate: fade-in

   The following section is for **ADVANCED USERS** who wants to extends the features of a 🍱.
   If you are not familiar with Jinja2, or you are not sure whether this is useful for you, we recommend you to skip this section. 
   While Jinja2 is a very powerful template engine that enables BentoML to support
   such features, we notice that this might be a little bit complicated for
   beginners.

   While this section fully exposes the generation process of the Dockerfile
   for a Bento, our recommendation is to just keep it simple. Sometime less is
   more 😃.

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
This allows BentoML to enable users to fully customize how to structure a 🍱's Dockerfile.

.. note::

    To use a custom Dockerfile template, users have to provide a file with a format
    that follows the Jinja2 template syntax. The template file should have
    extensions of :code:`.j2`, :code:`.template`, :code:`.jinja`.

In addition to Jinja, BentoML also uses `Docker Buildx <https://docs.docker.com/desktop/multi-arch/>`_, which enables users to build Bentos that support
multiple architectures. Under the hood the generated Dockefile leverage
`buildkit <https://github.com/moby/buildkit>`_, which enables some advanced `dockerfile features <https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md#dockerfile-frontend-syntaxes>`_.

.. note::

   Currently the only buildkit frontend that we are supporting is
   :code:`docker/dockerfile:1.4-labs`. Please contact us if you need support for
   different build frontend.


Writing custom Dockerfile template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To construct a custom :code:`Dockerfile` template, users have to provide the block below at the beginning of the Dockerfile template :code:`Dockerfile.template`:

.. code-block:: jinja

    {% extends bento__dockerfile %}

By doing so, we ensure that the generated Dockerfile will be compatible with a Bento.

.. dropdown:: About writing :code:`Dockerfile.template`
   :icon: code

   The Dockerfile template is a mix between :code:`Jinja2` syntax and :code:`Dockerfile`
   syntax. BentoML set both `trim_blocks` and `lstrip_blocks` in Jinja
   templates environment to :code:`True`. 

   Make sure that your Dockerfile instruction is **unindented** as if you are writting a normal Dockerfile.

   Refers to `Jinja Whitespace Control <https://jinja.palletsprojects.com/en/3.1.x/templates/#whitespace-control>`_.

   The following example is extracted from BentoML internal templates:

   .. code-block:: dockerfile

       {% block SETUP_BENTO_BASE_IMAGE %}
       FROM {{ __base_image__ }} as cached

       FROM {{ __base_image__ }} as base

           {% if not __user_defined_image__ %}
               {% for key in __supported_architectures__ if key not in __cached__ %}
                   {% if 'arm64' in key or 'aarch64' in key %}
                   {% do __cached__.append(key) %}
       FROM base as base-arm64
                   {% elif 'armhf' in key or 'armel' in key %}
                   {% do __cached__.append(key) %}
       FROM base as base-arm
                   {% elif 'i386' in key %}
                   {% do __cached__.append(key) %}
       FROM base as base-386
                   {% else %}
                   {% do __cached__.append(key) %}
       FROM base as base-{{ key }}
                   {% endif %}
               {% endfor %}
               {% else %}
       # custom base_image: {{ __base_image__ }}.
       # BentoML only support x86 architecture when using custom base_image
       FROM base as base-amd64
           {% endif %}

       FROM base-${TARGETARCH}

       ARG TARGETARCH

       ARG TARGETPLATFORM

       ENV LANG=C.UTF-8

       ENV LC_ALL=C.UTF-8

       ENV PYTHONIOENCODING=UTF-8

       ENV PYTHONUNBUFFERED=1

       ENV PATH {{ expands_bento_path("env", "docker", bento_path=bento__path) }}:/usr/local/bin:$PATH

       {% endblock %}

   .. note::
      Notice how for all Dockerfile instruction, we consider as if the Jinja
      logics aren't there 🚀.

Blocks
^^^^^^

BentoML defines a sets of `Blocks <https://jinja.palletsprojects.com/en/3.1.x/templates/#base-template>`_ under the object :code:`bento__dockerfile`.

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

Users are free to create,add their own block. However, we kindly ask users not
to create any new block with prefix :code:`SETUP_BENTO_*`. BentoML has
a internal validation process to make sure that the generated Dockerfile is able
to containerize a Bento.

To extend any given block, users can do so by adding :code:`{{ super() }}` at
any point inside block. This will ensure that the block is inherited from the
main :code:`SETUP_BENTO` block defined by BentoML.

The following are examples of how to use custom blocks:

.. tab-set::

    .. tab-item:: Example 1

        .. code:: jinja

           {% extends bento__dockerfile %}
           {% block SETUP_BENTO_USER %}
           {{ super() }}
           ENV CUSTOM_USER_VAR=foobar
           {% endblock %}

    .. tab-item:: Example 2

        .. code:: jinja

           {% extends bento__dockerfile %}
           {% block SETUP_BENTO_COMPONENTS %}
           RUN --mount=type=ssh git clone git@github.com:myorg/myproject.git myproject
           {{ super() }}
           {% endblock %}

    .. tab-item:: Example 3

        .. code:: jinja

           {% extends bento__dockerfile %}
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

An example of a custom Dockerfile template:

.. code-block:: jinja

    {% extends bento__dockerfile %}
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

        {% extends bento__dockerfile %}
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

.. tip::
    You might already noticed that there are also some variables that
    looks like :code:`{{ __base_image__ }}`. These given fields are not meant to
    be modified by users and should only be used for BentoML internal generation process.

Support functions
^^^^^^^^^^^^^^^^^

The following functions/class can be used to help with the Dockerfile generation process. 

.. warning::
    These are already used internally by BentoML and users are **NOT EXPECTED** to use them.
    They are here so that developers can use them if needed.

.. autofunction:: bentoml._internal.bento.gen.generate_dockerfile

.. autoclass:: bentoml._internal.bento.docker.DistroSpec
    :members:
