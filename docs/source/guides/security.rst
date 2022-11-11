========
Security
========

Securing Endpoint Access
------------------------

Server Side Authentication
^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable authentication for a given BentoServer endpoint, An authentication middleware can be added to :code:`bentoml.Service`'s via :code:`add_asgi_middleware` API. This API supports mounting
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

    svc.add_asgi_middleware(SessionMiddleware, secret_key='you_secret')


Certificates
^^^^^^^^^^^^

BentoML supports HTTPS with self-signed certificates. To enable HTTPS, you can to provide SSL certificate and key files as arguments
to the :code:`bentoml serve` command. Use :code:`bentoml serve --help` to see the full list of options.

.. code::
    
    bentoml serve iris_classifier:latest --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem


Reverse Proxy
^^^^^^^^^^^^^

It is a common practice to set up a reverse proxy server to handle rate limiting and authentication for any given backend services.


Service Mesh
^^^^^^^^^^^^

For Kubernetes users looking for advanced authentication, access control, and routing
policies, we recommend you to deploy Bentos with `Yatai <https://github.com/bentoml/Yatai>`_
and use Yatai's `Istio <https://istio.io/>`_ integration.



Security Policy
---------------

To report a vulnerability, we kindly ask you not to share it publicly on GitHub or in the community slack channel. Instead, contact the BentoML team directly at contact@bentoml.ai

View the full BentoML’s security policy `here <https://github.com/bentoml/BentoML/security/policy>`_.



.. TODO::

    * Base Image Security
    * Securing Yatai deployment
    * Reverse Proxy setup guide and sample code/config
    * Service Mesh setup guide and sample code/config
