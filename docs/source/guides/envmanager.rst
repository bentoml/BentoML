================
Environment Manager
================

*Not supported on Windows*

The Environment Manager creates an isolated environment based on the
dependencies from the bentofile.yaml to run bentoml-cli commands. This means you
can use the environment manager to create an environment and use it to run
``bentoml serve`` which will then serve your bento in the isolated environment
that is created. 

.. code-block:: bash

   » bentoml serve --env conda iris_classifier:latest

.. tip::

    Depending on the environment, the env-manager will use different tools
    behind the scenes to resolve and install the dependencies. The logs for
    these tools are hidden by env-manager default. To view them, run bentoml 
    in debug mode with ``--debug``/``--verbose`` flag.

Caching
~~~~~~~~~~~~

There are 2 types of environment that can be created for you based on the serving
target you invoke ``bentoml serve`` with. 
(checkout :ref:`bentoml serving reference <reference/cli:serve>` for more
information on different serving targets.)


1. Persistant environments - If the serving target is a Bento-Tag for a bento in
   the bento store, the created environment will be persistant and will be saved 
   to ``$BENTOML_HOME/env``. For subsequent invocations can use the same
   environment.
2. Ephimeral environments - In all other cases, ie if the serving target is the 
   import path of a ‘bentoml.Service’ instance or a folder containing a valid 
   ‘bentofile.yaml’ build file, the created environment will be removed after
   the bentoml command has been complete.

.. note::
   You can run ``rm -rf $BENTOML_HOME/env`` to clear the cache.
