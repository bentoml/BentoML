=================
Serving with gRPC
=================

This guide will demonstrate advanced features that BentoML offers for you to get started
with `gRPC <https://grpc.io/>`_:

- First-class support for :ref:`custom gRPC Servicer <guides/grpc:Mounting Servicer>`, :ref:`custom interceptors <guides/grpc:Mounting gRPC Interceptors>`, handlers.
- Seemlessly adding gRPC support to existing Bento.
- Streaming support (currently on our roadmap).

This guide will also walk your through some of the strengths and weaknesses of serving with gRPC, as well as
recommendation on scenarios where gRPC might be a good fit.

:bdg-info:`Requirements:` This guide assumes that you have basic knowledge of gRPC and protobuf. If you aren't
familar with gRPC, you can start with gRPC `quick start guide <https://grpc.io/docs/languages/python/quickstart/>`_.

.. seealso::

   For quick introduction to serving with gRPC, see :ref:`Intro to BentoML <tutorial:Tutorial: Intro to BentoML>`

Get started with gRPC in BentoML
--------------------------------

We will be using the example from :ref:`the quickstart<tutorial:Tutorial: Intro to BentoML>` to
demonstrate BentoML capabilities with gRPC.

Requirements
~~~~~~~~~~~~

Install BentoML with gRPC support with :pypi:`pip`:

.. code-block:: bash

   » pip install "bentoml[grpc]"

Thats it! You can now serving your Bento with gRPC via :ref:`bentoml serve-grpc <reference/cli:serve-grpc>` without having to modify your current service definition 😃.

.. code-block:: bash

   » bentoml serve-grpc iris_classifier:latest --production

Interact with your gRPC BentoService
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to interact with your gRPC BentoService:

1. Use tools such as :github:`fullstorydev/grpcurl`, :github:`fullstorydev/grpcui`: 
   The server requires :github:`reflection <grpc/grpc/blob/master/doc/server-reflection.md>` to be enabled for those tools to work.
   Pass in ``--enable-reflection`` to enable reflection:

   .. code-block:: bash

      » bentoml serve-grpc iris_classifier:latest --production --enable-reflection

   Open a different terminal and use one of the following:

   .. tab-set::

      .. tab-item:: gRPCurl

         We will use :github:`fullstorydev/grpcurl` to send a CURL-like request to the gRPC BentoServer.

         Note that we will use `docker <https://docs.docker.com/get-docker/>`_ to run the ``grpcurl`` command.

         .. tab-set::

            .. tab-item:: MacOS/Windows
               :sync: macwin

               .. code-block:: bash

                  » docker run -i --rm \
                               fullstorydev/grpcurl -d @ -plaintext host.docker.internal:3000 \
                               bentoml.grpc.v1alpha1.BentoService/Call <<EOT
                  {
                     "apiName": "classify",
                     "ndarray": {
                        "shape": [1, 4],
                        "floatValues": [5.9, 3, 5.1, 1.8]
                     }
                  }
                  EOT

            .. tab-item:: Linux
               :sync: Linux

               .. code-block:: bash

                  » docker run -i --rm \
                               --network=host \
                               fullstorydev/grpcurl -d @ -plaintext 0.0.0.0:3000 \
                               bentoml.grpc.v1alpha1.BentoService/Call <<EOT
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
               :sync: macwin

               .. code-block:: bash

                  » docker run --init --rm \
                               -p 8080:8080 fullstorydev/grpcui -plaintext host.docker.internal:3000

            .. tab-item:: Linux
               :sync: Linux

               .. code-block:: bash

                  » docker run --init --rm \
                               -p 8080:8080 \
                               --network=host fullstorydev/grpcui -plaintext 0.0.0.0:3000

         Proceed to http://127.0.0.1:8080 in your browser and send test request from the web UI.

2. Use one of the below :ref:`client implementations <guides/grpc:Client Implementation>` to send test requests to your BentoService.

Client Implementation
~~~~~~~~~~~~~~~~~~~~~

.. note::

   All of the following client implementations are :github:`available on GitHub <bentoml/BentoML/tree/main/docs/source/guides/snippets/grpc/>`.

.. note::

   Make sure to have ``protoc`` and any of the language-specifi plugin installed and
   aded to your ``PATH``.

   For example, if you are using Go, you will need to install ``protoc-gen-go-grpc``,
   and so on.

From another terminal, use one of the following client implementation to send request to the
gRPC server:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      We will create our Python client in the directory ``~/workspace/iris_python_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_python_client
         » cd ~/workspace/iris_python_client

      Now, create a file named ``client.py`` in the ``~/workspace/iris_python_client`` directory with the following content:

      .. literalinclude:: ./snippets/grpc/python/client.py
         :language: py
         :caption: `client.py`

   .. tab-item:: Go
      :sync: golang

      :bdg-info:`Requirements:` Make sure to install the `prerequisites <https://grpc.io/docs/languages/go/quickstart/#prerequisites>`_ before using Go.

      We will create our Golang client in the directory ``~/workspace/iris_go_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_go_client
         » cd ~/workspace/iris_go_client
         » go mod init iris_go_client && go mod tidy

      Add the following lines to ``~/workspace/iris_go_client/go.mod``:

      .. code-block:: go

         require github.com/bentoml/bentoml/grpc/v1alpha1 v0.0.0-unpublished

         replace github.com/bentoml/bentoml/grpc/v1alpha1 v0.0.0-unpublished => ./github.com/bentoml/bentoml/grpc/v1alpha1

      This is to make sure Go will know where to import our generated gRPC stubs from
      the incoming steps. (since we don't host the generate gRPC stubs on `pkg.go.dev` 😄)

      .. include:: ./snippets/grpc/docs/additional_setup.rst

      Given the your ``client.go`` will be saved under ``~/workspace/iris_go_client/client.go``, use the
      following ``protoc`` command to generate the gRPC Go stubs:

      .. code-block:: bash

         » protoc -I. -I thirdparty/protobuf/src  \
                  --go_out=. --go_opt=paths=import \
                  --go-grpc_out=. --go-grpc_opt=paths=import \
                  bentoml/grpc/v1alpha1/service.proto

      The following command is a hack to make the generated Go code importable:

      .. code-block:: bash

         » pushd github.com/bentoml/bentoml/grpc/v1alpha1
         » go mod init v1alpha1 && go mod tidy
         » popd

      Now, create a file named ``client.go`` in the ``~/workspace/iris_go_client`` directory with the following content:

      .. literalinclude:: ./snippets/grpc/go/client.go
         :language: go
         :caption: `client.go`

   .. tab-item:: C++
      :sync: cpp

      :bdg-info:`Requirements:` Make sure follow the `instructions <https://grpc.io/docs/languages/cpp/quickstart/#install-grpc>`_ to install gRPC and Protobuf locally.

      In the C++ world, there are various of different build tools. Feel free to use the one
      you are familiar with. In this example, we will use `bazel <https://bazel.build/>`_ to build and run the client.

      We will create our C++ client in the directory ``~/workspace/iris_cc_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_cc_client
         » cd ~/workspace/iris_cc_client

      Create a ``~/workspace/iris_cc_client/client.cpp`` file with the following content:

      .. literalinclude:: ./snippets/grpc/cpp/client.cc
         :language: cpp
         :caption: `client.cpp`

      Define a ``WORKSPACE`` file in the ``~/workspace/iris_cc_client`` directory:

      .. dropdown:: ``~/workspace/iris_cc_client/WORKSPACE``

         .. literalinclude:: ./snippets/grpc/cpp/WORKSPACE.snippet.bzl
            :language: python

      Then followed by defining a ``BUILD`` file:

      .. dropdown:: ``~/workspace/iris_cc_client/BUILD``

         .. literalinclude:: ./snippets/grpc/cpp/BUILD.snippet.bzl
            :language: python

   .. tab-item:: Java
      :sync: java

      :bdg-info:`Requirements:` Make sure to have `JDK>=7 <https://jdk.java.net/>`_ and follow the :github:`instructions <grpc/grpc-java/tree/master/compiler>` to install ``protoc`` plugin for gRPC Java.

      Additionally, you will need to also have :github:`gRPC Java <grpc/grpc-java>` and :github:`protocolbuffers/protobuf` available locally.

      We will create our Java client in the directory ``~/workspace/iris_java_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_java_client
         » cd ~/workspace/iris_java_client
         » mkdir -p src/main/java/com/client

      Feel free to use any Java build tools of choice (Maven, Gradle, Bazel, etc.) to build and run the client.

      In this example, we will use `bazel <bazel.build>`_ to build and run the client.

      Define a ``WORKSPACE`` file in the ``~/workspace/iris_java_client`` directory:

      .. dropdown:: ``~/workspace/iris_java_client/WORKSPACE``

         .. literalinclude:: ./snippets/grpc/java/WORKSPACE.snippet.bzl
            :language: python

      Then followed by defining a ``BUILD`` file:

      .. dropdown:: ``~/workspace/iris_java_client/BUILD``

         .. literalinclude:: ./snippets/grpc/java/BUILD.snippet.bzl
            :language: python

      .. include:: ./snippets/grpc/docs/additional_setup.rst

      Here is the ``protoc`` command to generate the gRPC Java stubs:

      .. code-block:: bash

         » protoc -I . \
                  -I ./thirdparty/protobuf/src \
                  --java_out=./src/main/java \
                  --grpc-java_out=./src/main/java \
                  bentoml/grpc/v1alpha1/service.proto

      Proceed to create a ``src/main/java/com/client/BentoServiceClient.java`` file with the following content:

      .. literalinclude:: ./snippets/grpc/java/src/main/java/com/client/BentoServiceClient.java
         :language: java
         :caption: `BentoServiceClient.java`

   .. tab-item:: Kotlin
      :sync: kotlin

      :bdg-info:`Requirements:` Make sure to have the `prequisites <https://grpc.io/docs/languages/kotlin/quickstart/#prerequisites>`_ to get started with :github:`grpc/grpc-kotlin`.

      Additionally, you will need to also install :github:`Kotlin gRPC codegen <grpc/grpc-kotlin/blob/master/compiler/README.md>` in order to generate gRPC stubs.

      To bootstrap the Kotlin client, feel free to use either `gradle <https://gradle.org/>`_ or
      `maven <https://maven.apache.org/>`_ to build and run the following client code.

      In this example, we will use `bazel <bazel.build>`_ to build and run the client.

      We will create our Kotlin client in the directory ``~/workspace/iris_kotlin_client/``, followed by creating the client directory structure:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_kotlin_client
         » cd ~/workspace/iris_kotlin_client
         » mkdir -p src/main/kotlin/com/client

      Define a ``WORKSPACE`` file in the ``~/workspace/iris_kotlin_client`` directory:

      .. dropdown:: ``~/workspace/iris_kotlin_client/WORKSPACE``

         .. literalinclude:: ./snippets/grpc/kotlin/WORKSPACE.snippet.bzl
            :language: python

      Then followed by defining a ``BUILD`` file:

      .. dropdown:: ``~/workspace/iris_kotlin_client/BUILD``

         .. literalinclude:: ./snippets/grpc/kotlin/BUILD.snippet.bzl
            :language: python

      .. include:: ./snippets/grpc/docs/additional_setup.rst

      Here is the ``protoc`` command to generate the gRPC Kotlin stubs:

      .. code-block:: bash

         » protoc -I. -I ./thirdparty/protobuf/src \
                  --kotlin_out ./kotlin/src/main/kotlin/ \
                  --grpc-kotlin_out ./kotlin/src/main/kotlin \
                  bentoml/grpc/v1alpha1/service.proto

      Proceed to create a ``src/main/kotlin/com/client/BentoServiceClient.kt`` file with the following content:

      .. literalinclude:: ./snippets/grpc/kotlin/src/main/kotlin/com/client/BentoServiceClient.kt
         :language: kotlin
         :caption: `BentoServiceClient.kt`

   .. tab-item:: Node.js
      :sync: js

      :bdg-info:`Requirements:` Make sure to have `Node.js <https://nodejs.org/en/>`_
      installed in your system.

      We will create our Node.js client in the directory ``~/workspace/iris_node_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_node_client
         » cd ~/workspace/iris_node_client

      Initialize the project and use the following ``package.json``:

      .. literalinclude:: ./snippets/grpc/node/package.json
         :language: json
         :caption: `package.json`

      Install the dependencies with either ``npm`` or ``yarn``:

      .. code-block:: bash

         » yarn install --add-devs

      .. note::

         If you are using M1, you might also have to prepend ``npm_config_target_arch=x64`` to ``yarn`` command:

         .. code-block:: bash

            » npm_config_target_arch=x64 yarn install --add-devs

      .. include:: ./snippets/grpc/docs/additional_setup.rst

      Here is the ``protoc`` command to generate the gRPC Javascript stubs:

      .. code-block:: bash

         » $(npm bin)/grpc_tools_node_protoc \
                  -I. --js_out=import_style=commonjs,binary:. \
                  --grpc_out=grpc_js:js \
                  bentoml/grpc/v1alpha1/service.proto

      Proceed to create a ``client.js`` file with the following content:

      .. literalinclude:: ./snippets/grpc/node/client.js
         :language: javascript
         :caption: `client.js`

   .. tab-item:: Swift
      :sync: swift

      :bdg-info:`Requirements:` Make sure to follow the :github:`instructions <grpc/grpc-swift/blob/main/docs/quick-start.md#prerequisites>` to install ``protoc`` plugin for gRPC Swift.

      Additionally, you will need to also have :github:`gRPC Swift <grpc/grpc-swift>` and :github:`protocolbuffers/protobuf` available locally.

      We will create our Swift client in the directory ``~/workspace/iris_swift_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_swift_client
         » cd ~/workspace/iris_swift_client

      Create a new Swift package:

      .. code-block:: bash

         » swift package init --name BentoServiceClient

      .. dropdown:: An example ``Package.swift`` that should be able to get you started:
         :icon: code

         .. literalinclude:: ./snippets/grpc/swift/Package.swift
            :language: swift

      .. include:: ./snippets/grpc/docs/additional_setup.rst

      Here is the ``protoc`` command to generate the gRPC swift stubs:

      .. code-block:: bash

         » protoc -I . -I ./thirdparty/protobuf/src \
                  --swift_out=Source/ --swift_opt=Visibility=Public \
                  --grpc-swift_out=Source/ --grpc-swift_opt=Visibility=Public \
                  bentoml/grpc/v1alpha1/service.proto

      Proceed to create a ``Sources/BentoServiceClient/main.swift`` file with the following content:

      .. literalinclude:: ./snippets/grpc/swift/Sources/BentoServiceClient/main.swift
         :language: swift
         :caption: `main.swift`

   .. tab-item:: .NET
      :sync: dotnet

      :bdg-primary:`Note:` Please check out the :github:`examples <grpc/grpc-dotnet/tree/master/examples>` folder for client implementation :github:`grpc/grpc-dotnet`

   .. tab-item:: Dart
      :sync: dart

      :bdg-primary:`Note:` Please check out the :github:`examples <grpc/grpc-dart/tree/master/examples>` folder for client implementation :github:`grpc/grpc-dart`


:raw-html:`<br />`

Then you can proceed to run the client scripts:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: bash

         » python -m client

   .. tab-item:: Go
      :sync: golang

      .. code-block:: bash

         » go run ./client.go

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: bash

         » bazel run :client_cc

      .. note::

         See the :github:`instructions on GitHub <bentoml/BentoML/tree/main/docs/source/guides/snippets/grpc/README.md>` for working C++ client.

   .. tab-item:: Java
      :sync: java

      .. code-block:: bash

         » bazel run :client_java

      .. note::

         See the :github:`instructions on GitHub <bentoml/BentoML/tree/main/docs/source/guides/snippets/grpc/README.md>` for working Java client.

   .. tab-item:: Kotlin
      :sync: kotlin

      .. code-block:: bash

         » bazel run :client_kt

      .. note::

         See the :github:`instructions on GitHub <bentoml/BentoML/tree/main/docs/source/guides/snippets/grpc/README.md>` for working Kotlin client.

   .. tab-item:: Node.js
      :sync: js

      .. code-block:: bash

         » node client.js

   .. tab-item:: Swift
      :sync: swift

      .. code-block:: bash

         » swift run BentoServiceClient

   .. tab-item:: .NET
      :sync: dotnet

      :bdg-primary:`Note:` Please check out the :github:`examples <grpc/grpc-dotnet/tree/master/examples>` folder for client implementation :github:`grpc/grpc-dotnet`

   .. tab-item:: Dart
      :sync: dart

      :bdg-primary:`Note:` Please check out the :github:`examples <grpc/grpc-dart/tree/master/examples>` folder for client implementation :github:`grpc/grpc-dart`


After successfully running the client, proceed to build the bento as usual:

.. code-block:: bash

   » bentoml build

:raw-html:`<br />`

Containerize your Bento 🍱 with gRPC support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To containerize the Bento with gRPC features, pass in ``--enable-features=grpc`` to
:ref:`bentoml containerize <reference/cli:containerize>` to add additional gRPC
dependencies to your Bento

.. code-block:: bash

   » bentoml containerize iris_classifier:latest --enable-features=grpc

``--enable-features`` allows users to containerize any of the existing Bentos with :ref:`additional features </installation:Additional features>` without having to rebuild the Bento.

.. note::

   ``--enable-features`` accepts a comma-separated list of features or multiple arguments.

After containerization, your Bento container can now be used with gRPC:

.. code-block:: bash

   » docker run -it --rm \
                -p 3000:3000 -p 3001:3001 \
                iris_classifier:6otbsmxzq6lwbgxi serve-grpc --production

Congratulations! You have successfully served, containerized and tested your BentoService with gRPC.

:raw-html:`<br />`

Using gRPC in BentoML
---------------------

Protobuf definition
~~~~~~~~~~~~~~~~~~~

Let's take a quick look at `protobuf <https://developers.google.com/protocol-buffers/>`_  definition of the BentoService:

.. code-block:: protobuf

   service BentoService {
     rpc Call(Request) returns (Response) {}
   }

.. dropdown:: `Expands for current protobuf definition.`
   :icon: code

   .. tab-set::

      .. tab-item:: v1alpha1

         .. literalinclude:: ../../../bentoml/grpc/v1alpha1/service.proto
            :language: protobuf

As you can see, BentoService defines a `simple rpc` ``Call`` that sends a ``Request`` message and returns a ``Response`` message.

A ``Request`` message takes in:

* `api_name`: the name of the API function defined inside your BentoService. 
* `oneof <https://developers.google.com/protocol-buffers/docs/proto3#oneof>`_ `content`: the field can be one of the following types:

+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| Protobuf definition                                              | IO Descriptor                                                                             |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| :ref:`guides/grpc:Array representation via ``NDArray```          | :ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`       |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| :ref:`guides/grpc:Tabular data representation via ``DataFrame``` | :ref:`bentoml.io.PandasDataFrame <reference/api_io_descriptors:Tabular Data with Pandas>` |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| :ref:`guides/grpc:File-like object via ``File```                 | :ref:`bentoml.io.File <reference/api_io_descriptors:Files>`                               |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| |google_protobuf_string_value|_                                  | :ref:`bentoml.io.Text <reference/api_io_descriptors:Texts>`                               |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| |google_protobuf_value|_                                         | :ref:`bentoml.io.JSON <reference/api_io_descriptors:Structured Data with JSON>`           |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| :ref:`guides/grpc:Complex payload via ``Multipart```             | :ref:`bentoml.io.Multipart <reference/api_io_descriptors:Multipart Payloads>`             |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| :ref:`guides/grpc:Compact data format via ``serialized_bytes```  | (See below)                                                                               |
+------------------------------------------------------------------+-------------------------------------------------------------------------------------------+

.. note::

   ``Series`` is currently not yet supported.

.. _google_protobuf_value: https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#google.protobuf.Value

.. |google_protobuf_value| replace:: ``google.protobuf.Value``

.. _google_protobuf_string_value: https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#stringvalue

.. |google_protobuf_string_value| replace:: ``google.protobuf.StringValue``

The ``Response`` message will then return one of the aforementioned types as result.

:raw-html:`<br />`

:bdg-info:`Example:` In the :ref:`quickstart guide<tutorial:Creating a Service>`, we defined a ``classify`` API that takes in a :ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`.

Therefore, our ``Request`` message would have the following structure:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         req = pb.Request(
            api_name="classify",
            ndarray=pb.NDArray(
               dtype=pb.NDArray.DTYPE_FLOAT, shape=(1, 4), float_values=[5.9, 3, 5.1, 1.8]
            ),
         )

   .. tab-item:: Go
      :sync: golang

      .. code-block:: go

         req := &pb.Request{
            ApiName: "classify",
            Content: &pb.Request_Ndarray{
               Ndarray: &pb.NDArray{
                  Dtype: *pb.NDArray_DTYPE_FLOAT.Enum(),
                  Shape: []int32{1, 4},
                  FloatValues: []float32{3.5, 2.4, 7.8, 5.1}
               }
            }
         }

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "bentoml/grpc/v1alpha1/service.pb.h"

         using bentoml::grpc::v1alpha1::BentoService;
         using bentoml::grpc::v1alpha1::NDArray;
         using bentoml::grpc::v1alpha1::Request;

         std::vector<float> data = {3.5, 2.4, 7.8, 5.1};
         std::vector<int> shape = {1, 4};

         Request request;
         request.set_api_name("classify");

         NDArray *ndarray = request.mutable_ndarray();
         ndarray->mutable_shape()->Assign(shape.begin(), shape.end());
         ndarray->mutable_float_values()->Assign(data.begin(), data.end());

Array representation via ``NDArray``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:bdg-info:`Description:` ``NDArray`` represents a flattened n-dimensional array of arbitrary type.

``NDArray`` accepts

:bdg-primary:`API reference:` :ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`

Tabular data representation via ``DataFrame``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File-like object via ``File``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:bdg-info:`Description:` ``File`` represents any arbitrary file type. this can be used
to 

``File`` is a special type of ``Request`` content that allows users to send files to the BentoService.

Complex payload via ``Multipart``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compact data format via ``serialized_bytes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We introduce the field ``serialized_bytes`` to both ``Request`` and ``Response`` such
that the payload is serialized with BentoML's internal serialization format.

This is useful to when we want to send a large amount of data on the wire.
However, as mentioned above, this is an internal serialization format and thus not
**recommended** for use by users.

add me

Mounting Servicer
~~~~~~~~~~~~~~~~~

With support for :ref:`multiplexing <guides/grpc:Demystifying the misconception of gRPC vs. REST>`
to eliminate :wiki:`head-of-line blocking <Head-of-line_blocking>`,
gPRC enables us to mount additional custom servicess alongside with BentoService,
and serve them under the same port.

.. code-block:: python
   :caption: `service.py`
   :emphasize-lines: 13

   import route_guide_pb2
   import route_guide_pb2_grpc
   from servicer_impl import RouteGuideServicer

   svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

   services_name = [
       v.full_name for v in route_guide_pb2.DESCRIPTOR.services_by_name.values()
   ]
   svc.mount_grpc_servicer(
       RouteGuideServicer,
       add_servicer_fn=add_RouteGuideServicer_to_server,
       service_names=services_name,
   )

Serve your service with :ref:`bentoml serve-grpc <reference/cli:serve-grpc>` command:

.. code-block:: bash

   » bentoml serve-grpc service.py:svc --reload --enable-reflection

Now your ``RouteGuide`` service can also be accessed through ``localhost:3000``.

.. note::

   ``service_names`` is **REQUIRED** here, as this will be used for :github:`server reflection <grpc/grpc/blob/master/doc/server-reflection.md>`
   when ``--enable-reflection`` is passed to ``bentoml serve-grpc``.

Mounting gRPC Interceptors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Inteceptors are a component of gRPC that allows us to intercept and interact with the
proto message and service context either before - or after - the actual RPC call was
sent/received by client/server.

Interceptors to gRPC is what middleware is to HTTP. The most common use-case for interceptors
are authentication, :ref:`tracing <guides/tracing:Tracing>`, access logs, and more.

BentoML comes with a sets of built-in *async interceptors* to provide support for access logs,
`OpenTelemetry <https://opentelemetry.io/>`_, and `Prometheus <https://prometheus.io/>`_.

The following diagrams demonstrates the flow of a gRPC request from client to server:

.. image:: /_static/img/interceptor-flow.svg
   :alt: Interceptor Flow

Since interceptors are executed in the order they are added, users interceptors will be executed after the built-in interceptors.

Users interceptors should be **READ-ONLY**, which means it shouldn't modify the state or content of the incoming ``Request``.

BentoML currently only support **async interceptors** (via `grpc.aio.ServerInterceptor <https://grpc.github.io/grpc/python/grpc_asyncio.html#grpc.aio.ServerInterceptor>`_, as opposed to `grpc.ServerInterceptor <https://grpc.github.io/grpc/python/grpc_asyncio.html#grpc.aio.ServerInterceptor>`_). This is
because BentoML gRPC server is an async implementation of gRPC server.

.. note::

   If you are using ``grpc.ServerInterceptor``, you will need to migrate it over
   to use the new ``grpc.aio.ServerInterceptor`` in order to use this feature.

   Feel free to reach out to us at `#support on Slack <https://l.linklyhq.com/l/ktOX>`_

.. dropdown:: A toy implementation ``AppendMetadataInterceptor``

   .. code-block:: python
      :caption: metadata_interceptor.py

      from __future__ import annotations

      import typing as t
      import functools
      import dataclasses
      from typing import TYPE_CHECKING

      from grpc import aio

      if TYPE_CHECKING:
          from bentoml.grpc.types import Request
          from bentoml.grpc.types import Response
          from bentoml.grpc.types import RpcMethodHandler
          from bentoml.grpc.types import AsyncHandlerMethod
          from bentoml.grpc.types import HandlerCallDetails
          from bentoml.grpc.types import BentoServicerContext


      @dataclasses.dataclass
      class Context:
          usage: str
          accuracy_score: float


      class AppendMetadataInterceptor(aio.ServerInterceptor):
           def __init__(self, *, usage: str, accuracy_score: float) -> None:
               self.context = Context(usage=usage, accuracy_score=accuracy_score)
               self._record: set[str] = set()

           async def intercept_service(
               self,
               continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
               handler_call_details: HandlerCallDetails,
           ) -> RpcMethodHandler:
               from bentoml.grpc.utils import wrap_rpc_handler

               handler = await continuation(handler_call_details)

               if handler and (handler.response_streaming or handler.request_streaming):
                   return handler

               def wrapper(behaviour: AsyncHandlerMethod[Response]):
                   @functools.wraps(behaviour)
                   async def new_behaviour(
                      request: Request, context: BentoServicerContext
                   ) -> Response | t.Awaitable[Response]:
                       self._record.update(
                         {f"{self.context.usage}:{self.context.accuracy_score}"}
                       )
                       resp = await behaviour(request, context)
                       context.set_trailing_metadata(
                          tuple(
                                [
                                   (k, str(v).encode("utf-8"))
                                   for k, v in dataclasses.asdict(self.context).items()
                                ]
                          )
                       )
                       return resp

                   return new_behaviour

               return wrap_rpc_handler(wrapper, handler)

To add your intercptors to existing BentoService, use ``svc.add_grpc_interceptor``:

.. code-block:: python
   :caption: `service.py`

   from custom_interceptor import CustomInterceptor

   svc.add_grpc_interceptor(CustomInterceptor)

.. note::

   ``add_grpc_interceptor`` also supports `partial` class as well as multiple arguments
   interceptors:

   .. tab-set::

      .. tab-item:: multiple arguments

         .. code-block:: python

            from metadata_interceptor import AppendMetadataInterceptor

            svc.add_grpc_interceptor(AppendMetadataInterceptor, usage="NLP", accuracy_score=0.867)

      .. tab-item:: partial method

         .. code-block:: python

            from functools import partial

            from metadata_interceptor import AppendMetadataInterceptor

            svc.add_grpc_interceptor(partial(AppendMetadataInterceptor, usage="NLP", accuracy_score=0.867))

--------------

Recommendations
---------------

gRPC is designed to be high performance framework for inter-service communications. This
means that it is a perfect fit for building microservices. The following are some
recommendation we have for using gRPC for model serving:

:raw-html:`<br />`

Demystifying the misconception of gRPC vs. REST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You might stumble upon articles comparing gRPC to REST, and you might get the impression
that gRPC is a better choice than REST when building services. This is not entirely
true.

gRPC is built on top of HTTP/2, and it addresses some of the shortcomings of HTTP/1.1,
such as :wiki:`head-of-line blocking <Head-of-line_blocking>`, and :wiki:`HTTP pipelining <HTTP_pipelining>`.
However, gRPC is not a replacement for REST, and indeed it is not a replacement for
model serving. gRPC comes with its own set of trade-offs, such as:

* **Limited browser support**: It is impossible to call a gRPC service directly from any
  browser. You will end up using tools such as :github:`gRPCUI <fullstorydev/grpcui>` in order to interact
  with your service, or having to go through the hassle of implementing a gRPC client in
  your language of choice.

* **Binary protocol format**: While :github:`Protobuf <protocolbuffers/protobuf>` is
  efficient to send and receive over the wire, it is not human-readable. This means
  additional toolin for debugging and analyzing protobuf messages are required.

* **Knowledge gap**: gRPC comes with its own concepts and learning curve, which requires
  teams to invest time in filling those knowledge gap to be effectively use gRPC. This
  often leads to a lot of friction and sometimes increase friction to the development
  agility.

* **Lack of suport for additional content types**: gRPC depends on protobuf, its content
  type are restrictive, in comparison to out-of-the-box support from HTTP+REST.

.. seealso::

   `gRPC on HTTP/2 <https://grpc.io/blog/grpc-on-http2/>`_ dives into how gRPC is built
   on top of HTTP/2, and this `article <https://www.cncf.io/blog/2018/07/03/http-2-smarter-at-scale/>`_
   goes into more details on how HTTP/2 address the problem from HTTP/1.1

   For HTTP/2 specification, see `RFC 7540 <https://tools.ietf.org/html/rfc7540>`_.

:raw-html:`<br />`

Should I use gRPC instead of REST for model serving?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes and no.

If your organization is already using gRPC for inter-service communications, using
your Bento with gRPC is a no-brainer. You will be able to seemlessly integrate your
Bento with your existing gRPC services without having to worry about the overhead of
implementing :github:`grpc-gateway <grpc-ecosystem/grpc-gateway>`.

However, if your organization is not using gRPC, we recommend to keep using REST for
model serving. This is because REST is a well-known and well-understood protocol,
meaning there is no knowledge gap for your team, which will increase developer agility, and
faster go-to-market strategy.

:raw-html:`<br />`

Performance tuning
~~~~~~~~~~~~~~~~~~

BentoML allows user to tune the performance of gRPC via :ref:`bentoml_configuration.yaml <guides/configuration:Configuring BentoML>` via ``api_server.grpc``.

A quick overview of the available configuration for gRPC:

.. code-block:: yaml
   :caption: `bentoml_configuration.yaml`

   api_server:
     grpc:
       host: 0.0.0.0
       port: 3000
       max_concurrent_streams: ~
       maximum_concurrent_rpcs: ~
       max_message_length: -1
       reflection:
         enabled: false
       metrics:
         host: 0.0.0.0
         port: 3001

:raw-html:`<br />`

``max_concurrent_streams``
^^^^^^^^^^^^^^^^^^^^^^^^^^

   :bdg-info:`Definition:` Maximum number of concurrent incoming streams to allow on a HTTP2 connection.

By default we don't set a limit cap. HTTP/2 connections typically has limit of `maximum concurrent streams <httpwg.org/specs/rfc7540.html#rfc.section.5.1.2>`_
on a connection at one time.

.. dropdown:: Some notes about fine-tuning ``max_concurrent_streams``

   Note that a gRPC channel uses a single HTTP/2 connection, and concurrent calls are multiplexed on said connection.
   When the number of active calls reaches the connection stream limit, any additional
   calls are queued to the client. Queued calls then wait for active calls to complete before being sent. This means that
   application will higher load and long running streams could see a performance degradation caused by queuing because of the limit.

   Setting a limit cap on the number of concurrent streams will prevent this from happening, but it also means that
   you need to tune the limit cap to the right number. 

   * If the limit cap is too low, you will sooner or later running into the issue mentioned above.

   * Not setting a limit cap are also **NOT RECOMMENDED**. Too many streams on a single
     HTTP/2 connection introduces `thread contention` between streams trying to write
     to the connection, `packet loss` which causes all call to be blocked.

   :bdg-info:`Remarks:` We recommend you to play around with the limit cap, starting with 100, and increase if needed.

:raw-html:`<br />`

``maximum_concurrent_rpcs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

   :bdg-info:`Definition:` The maximum number of concurrent RPCs this server will service before returning ``RESOURCE_EXHAUSTED`` status.

By default we set to ``None`` to indicate no limit, and let gRPC to decide the limit.

:raw-html:`<br />`

``max_message_length``
^^^^^^^^^^^^^^^^^^^^^^

   :bdg-info:`Definition:` The maximum message length in bytes allowed to be received on/can be send to the server.

By default we set to ``-1`` to indicate no limit.
Message size limits via this options is a way to prevent gRPC from consuming excessive
resources. By default, gRPC uses per-message limits to manage inbound and outbound
message.

.. dropdown:: Some notes about fine-tuning ``max_message_length``

   This options sets two values: :github:`grpc.max_receive_message_length <grpc/grpc/blob/e8df8185e521b518a8f608b8a5cf98571e2d0925/include/grpc/impl/codegen/grpc_types.h#L153>`
   and :github:`grpc.max_send_message_length <grpc/grpc/blob/e8df8185e521b518a8f608b8a5cf98571e2d0925/include/grpc/impl/codegen/grpc_types.h#L159>`.

   .. code-block:: cpp

      #define GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH "grpc.max_receive_message_length"

      #define GRPC_ARG_MAX_SEND_MESSAGE_LENGTH "grpc.max_send_message_length"

   By default, gRPC sets incoming message to be 4MB, and no limit on outgoing message.
   We recommend you to only set this option if you want to limit the size of outcoming message. Otherwise, you should let gRPC to determine the limit.


We recommend you to also check out `gRPC performance best practice <https://grpc.io/docs/guides/performance/>`_ to learn about best practice for gRPC.

