.. tab-set::

   .. tab-item:: gRPCurl

      We will use :github:`fullstorydev/grpcurl` to send a CURL-like request to the gRPC BentoServer.

      Note that we will use `docker <https://docs.docker.com/get-docker/>`_ to run the ``grpcurl`` command.

      .. tab-set::

         .. tab-item:: MacOS/Windows
            :sync: __fullstorydev_macwin

            .. code-block:: bash

               » docker run -i --rm \
                              fullstorydev/grpcurl -d @ -plaintext host.docker.internal:3000 \
                              bentoml.grpc.v1.BentoService/Call <<EOT
               {
                  "apiName": "classify",
                  "ndarray": {
                     "shape": [1, 4],
                     "floatValues": [5.9, 3, 5.1, 1.8]
                  }
               }
               EOT

         .. tab-item:: Linux
            :sync: __fullstorydev_linux

            .. code-block:: bash

               » docker run -i --rm \
                              --network=host \
                              fullstorydev/grpcurl -d @ -plaintext 0.0.0.0:3000 \
                              bentoml.grpc.v1.BentoService/Call <<EOT
               {
                  "apiName": "classify",
                  "ndarray": {
                     "shape": [1, 4],
                     "floatValues": [5.9, 3, 5.1, 1.8]
                  }
               }
               EOT

   .. tab-item:: gRPCUI

      We will use :github:`fullstorydev/grpcui` to send request from a web browser.

      Note that we will use `docker <https://docs.docker.com/get-docker/>`_ to run the ``grpcui`` command.

      .. tab-set::

         .. tab-item:: MacOS/Windows
            :sync: __fullstorydev_macwin

            .. code-block:: bash

               » docker run --init --rm \
                              -p 8080:8080 fullstorydev/grpcui -plaintext host.docker.internal:3000

         .. tab-item:: Linux
            :sync: __fullstorydev_linux

            .. code-block:: bash

               » docker run --init --rm \
                              -p 8080:8080 \
                              --network=host fullstorydev/grpcui -plaintext 0.0.0.0:3000

      Proceed to http://127.0.0.1:8080 in your browser and send test request from the web UI.

