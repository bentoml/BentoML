Deploy yatai server behind NGINX
================================

The control service of yatai server is currently using insecure gRPC,
which is actually a HTTP/2 Cleartext (H2C) service. Normally:

.. code:: bash

    bentoml config set yatai_service.url=<ip-of-your-server>:50051

But for security or ease of management, we sometimes want to deploy it
behind an Nginx server, and use our own certificate to encrypt it.

To achieve this, we need to follow steps below.

1. make NGINX proxying HTTP/2 from h2 to h2c
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NGINX ≥ 1.13.9

``nginx.conf``

::

    ...
    http{
        server {
            listen [::]:1443 ssl http2 ipv6only=on;
            listen 1443 ssl http2;

            ssl_certificate /etc/letsencrypt/live/yatai.yourdomain.com/fullchain.pem;
            ssl_certificate_key /etc/letsencrypt/live/yatai.yourdomain.com/privkey.pem;

            location / {
                grpc_pass grpc://localhost:50051;
            }
        }

        server {  # additional config to proxy yatai dashboard as HTTPS
            server_name yatai.yourdomain.com;

            location / {
                proxy_pass http://127.0.0.1:3000;
            }

            listen [::]:443 ssl ipv6only=on;
            listen 443 ssl;
            ssl_certificate /etc/letsencrypt/live/yatai.yourdomain.com/fullchain.pem;
            ssl_certificate_key /etc/letsencrypt/live/yatai.yourdomain.com/privkey.pem;
            include /etc/letsencrypt/options-ssl-nginx.conf;
            ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
        }
    }

2. config yatai client to use h2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    bentoml config set yatai_service.url=grpcs://yatai.yourdomain.com:1443

3. (Optional) using self-signed certificates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    bentoml config set yatai_service.client_certificate_file=<path_to_your_ca_cert.pem>

More options of gRPC NGINX configuration:
`https://www.nginx.com/blog/nginx-1-13-10-grpc/ <https://www.nginx.com/blog/nginx-1-13-10-grpc/>`__

.. spelling::

    Cleartext
    proxying
