.. dropdown:: Additional setup
   :icon: gear

   We will also need to include the protobuf files from :github:`protocolbuffers/protobuf` as a thirdparty dependency:

   .. code-block:: bash

      » mkdir -p thirdparty && cd thirdparty
      » git clone --depth 1 https://github.com/protocolbuffers/protobuf.git

   Since we only requires the service proto file from :github:`bentoml/bentoml` to build the client, we will perform
   a `sparse checkout <https://github.blog/2020-01-17-bring-your-monorepo-down-to-size-with-sparse-checkout/>`_ to only checkout ``bentoml/grpc`` directory:

   .. code-block:: bash

      » mkdir bentoml && pushd bentoml
      » git init
      » git remote add -f origin https://github.com/bentoml/BentoML.git
      » git config core.sparseCheckout true
      » cat <<EOT >|.git/info/sparse-checkout
      bentoml/grpc
      EOT
      » git pull origin main
      » popd
