===================
Environment Manager
===================

:bdg-info:`Note:` This feature is currently only supported on UNIX/MacOS.

Environment manager is a utility that helps create an isolated environment to
run the BentoML CLI. Dependencies are pulled from your defined
``bentofile.yaml`` and the environment is built upon request. This means by
passing ``--env`` to supported CLI commands (such as :ref:`bentoml serve
<reference/cli:serve>`), such commands will then be run in an sandbox
environment that mimics the behaviour during production.

.. code-block:: bash

   » bentoml serve --env conda iris_classifier:latest

This creates and isolated conda environment from the dependencies in the bento
and runs ``bentoml serve`` from that environment.

.. note:: The current implementation will try to install the given dependencies
   before running the CLI command. Therefore, the environment startup will be a
   blocking call.


BentoML CLI Commands that support Environment Manager
    - :ref:`serve <reference/cli:serve>`
    - :ref:`serve-grpc <reference/cli:serve-grpc>`

Supported Environments
    - conda


Caching strategies
==================

Currently, there are two types of environments that are supported by the
environment manager:

1. Persistent environment: If the given target is a Bento, then the created
   environment will be stored locally to ``$BENTOML_HOME/env``. Such an
   environment will then be cached and later used by subsequent invocations.

2. Ephemeral environment: In cases where the given target is not a Bento (import
   path to ``bentoml.Service``, project directory containing a valid
   ``bentofile.yaml``), the environment will be created and cleanup up on
   demand.

.. note::
   You can run ``rm -rf $BENTOML_HOME/env`` to clear the cache.
