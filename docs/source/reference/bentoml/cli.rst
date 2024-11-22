===========
BentoML CLI
===========

BentoML CLI commands have usage documentation. You can learn more by running ``bentoml --help``. The ``--help`` flag also applies to sub-commands for viewing detailed usage of a command, like ``bentoml build --help``.

.. seealso::

   For details about managing BentoCloud Deployments using the BentoML CLI, see :doc:`/reference/bentocloud/bentocloud-cli`.

.. click:: bentoml_cli.bentos:bento_command
  :prog: bentoml
  :nested: full

.. click:: bentoml_cli.containerize:containerize_command
  :prog: bentoml containerize
  :nested: full

.. click:: bentoml_cli.env:env_command
  :prog: bentoml env
  :nested: full

.. click:: bentoml_cli.models:model_command
  :prog: bentoml models
  :nested: full

.. click:: bentoml_cli.serve:serve_command
  :prog: bentoml serve
  :nested: full
