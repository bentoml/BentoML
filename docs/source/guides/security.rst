========
Security
========

Securing Endpoint Access
------------------------

Server Side Authentication
^^^^^^^^^^^^^^^^^^^^^^^^^^

For authentication on the BentoServer endpoint itself, a simple way to do it is via
:code:`bentoml.Service`'s :code:`add_asgi_middleware` API. This API supports mounting
any ASGI middleware to the BentoServer endpoints. And many of the middlewares built by
the Python community, provides authentication or security functionality.

For example, you may apply HTTPS redirect and set trusted host URLs this way:

.. code::

    from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    svc = bentoml.Service('my_service', runners=[...])

    svc.add_asgi_middleware(TrustedHostMiddleware, allowed_hosts=['example.com', '*.example.com'])
    svc.add_asgi_middleware(HTTPSRedirectMiddleware)


For JWT authentication, check out the `starlette-authlib <https://github.com/aogier/starlette-authlib>`_
and `starlette-auth-toolkit <https://github.com/florimondmanca/starlette-auth-toolkit>`_.
Here's an example with starlette-authlib:

.. code::

    from starlette_authlib.middleware import AuthlibMiddleware as SessionMiddleware

    svc = bentoml.Service('my_service', runners=[...])

    svc.add_asgi_middleware(SessionMiddleware, secret='you_secret')


Reverse Proxy
^^^^^^^^^^^^^

It is more common to setup a reverse proxy server in front of a backend service, which
handles rate limiting and authentication.


Service Mesh
^^^^^^^^^^^^

For Kubernetes users looking for advanced authentication, access control, and routing
policies, we recommend deploying bentos with `Yatai <https://github.com/bentoml/Yatai>`_
and using Yatai's `Istio <https://istio.io/>`_ integration.



Security Policy
---------------

To report a vulnerability, please do not share it publicly on GitHub
or the community slack channel. Instead, contact the BentoML team
directly via email first: contact@bentoml.ai

View the full BentoML security policy `here <https://github.com/bentoml/BentoML/security/policy>`_.


.. TODO::

    * Base Image Security
    * Securing Yatai deployment
    * Reverse Proxy setup guide and sample code/config
    * Service Mesh setup guide and sample code/config
