==========
Components
==========

bentoml.Service
---------------

.. autoclass:: bentoml.Service
    :members: api, runners, apis, mount_asgi_app, mount_wsgi_app, add_asgi_middleware
    :undoc-members:

.. autofunction:: bentoml.load

.. TODO::
    Add docstring to the following classes/functions

bentoml.build
-------------

.. autofunction:: bentoml.bentos.build

.. autofunction:: bentoml.bentos.build_bentofile

.. autofunction:: bentoml.bentos.containerize


bentoml.Bento
-------------

.. autoclass:: bentoml.Bento
    :members: tag, info, path, path_of, doc
    :undoc-members:

bentoml.Runner
--------------

.. autoclass:: bentoml.Runner

bentoml.Runnable
----------------

.. autoclass:: bentoml.Runnable
    :members: method
    :undoc-members:

Tag
---

.. autoclass:: bentoml.Tag

Model
-----

.. autoclass:: bentoml.Model
    :members: to_runner, to_runnable, info, path, path_of, with_options
    :undoc-members:


YataiClient
-----------

.. autoclass:: bentoml.YataiClient
