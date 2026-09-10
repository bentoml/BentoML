===========
Server APIs
===========

The ``bentoml.server`` module provides lower-level server classes for existing code
that needs direct control over server startup and shutdown.

.. warning::

   The ``bentoml.server`` module is deprecated. For new code, use
   ``bentoml.serve()`` as shown in :doc:`/build-with-bentoml/clients`.

.. autoclass:: bentoml.server.HTTPServer
   :members: start, get_client, stop

.. autoclass:: bentoml.server.GrpcServer
   :members: start, get_client, stop
