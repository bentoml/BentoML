=================
Serving with gRPC
=================

*time expected: 12 minutes*

This guide will demonstrate advanced features that BentoML offers for you to get started
with `gRPC <https://grpc.io/>`_:

- First-class support for :ref:`custom gRPC Servicer <guides/grpc:Mounting Servicer>`, :ref:`custom interceptors <guides/grpc:Mounting gRPC Interceptors>`, handlers.
- Seemlessly adding gRPC support to existing Bento.

This guide will also walk you through tradeoffs of serving with gRPC, as well as
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

BentoML supports for gRPC are introduced in version 1.0.6 and above.

Install BentoML with gRPC support with :pypi:`pip`:

.. code-block:: bash

   » pip install -U "bentoml[grpc]"

Thats it! You can now serve your Bento with gRPC via :ref:`bentoml serve-grpc <reference/cli:serve-grpc>` without having to modify your current service definition 😃.

.. code-block:: bash

   » bentoml serve-grpc iris_classifier:latest

Using your gRPC BentoService
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to interact with your gRPC BentoService:

1. Use tools such as :github:`fullstorydev/grpcurl`, :github:`fullstorydev/grpcui`:
   The server requires :github:`reflection <grpc/grpc/blob/master/doc/server-reflection.md>` to be enabled for those tools to work.
   Pass in ``--enable-reflection`` to enable reflection:

   .. code-block:: bash

      » bentoml serve-grpc iris_classifier:latest --enable-reflection

   .. include:: ./snippets/grpc/grpc_tools.rst

   Open a different terminal and use one of the following:

2. Use one of the below :ref:`client implementations <guides/grpc:Client Implementation>` to send test requests to your BentoService.

.. _workspace: https://bazel.build/concepts/build-ref

.. |workspace| replace:: ``WORKSPACE``

.. _build: https://bazel.build/concepts/build-files

.. |build| replace:: ``BUILD``

.. _bazel: https://bazel.build

.. |bazel| replace:: `bazel`

Client Implementation
~~~~~~~~~~~~~~~~~~~~~

.. note::

   All of the following client implementations are :github:`available on GitHub <bentoml/BentoML/tree/main/grpc-client/>`.

:raw-html:`<br />`

From another terminal, use one of the following client implementation to send request to the
gRPC server:

.. note::

   gRPC comes with supports for multiple languages. In the upcoming sections
   we will demonstrate two workflows of generating stubs and implementing clients:

   - Using |bazel|_ to manage and isolate dependencies (recommended)
   - A manual approach using ``protoc`` its language-specific plugins

.. tab-set::

   .. tab-item:: Python
      :sync: python

      We will create our Python client in the directory ``~/workspace/iris_python_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_python_client
         » cd ~/workspace/iris_python_client

      Create a ``client.py`` file with the following content:

      .. literalinclude:: ../../../grpc-client/python/client.py
         :language: python
         :caption: `client.py`

   .. tab-item:: Go
      :sync: golang

      :bdg-info:`Requirements:` Make sure to install the `prerequisites <https://grpc.io/docs/languages/go/quickstart/#prerequisites>`_ before using Go.

      We will create our Golang client in the directory ``~/workspace/iris_go_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_go_client
         » cd ~/workspace/iris_go_client

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            Define a |workspace|_ file:

            .. dropdown:: ``WORKSPACE``
               :icon: code

               .. literalinclude:: ./snippets/grpc/go/WORKSPACE.snippet.bzl
                  :language: python

            Followed by defining a |build|_ file:

            .. dropdown:: ``BUILD``
               :icon: code

               .. literalinclude:: ./snippets/grpc/go/BUILD.snippet.bzl
                  :language: python

         .. tab-item:: Using protoc and language-specific plugins
            :sync: protoc-and-plugins

            Create a Go module:

            .. code-block:: bash

               » go mod init iris_go_client && go mod tidy

            Add the following lines to ``~/workspace/iris_go_client/go.mod``:

            .. code-block:: go

               require github.com/bentoml/bentoml/grpc/v1 v0.0.0-unpublished

               replace github.com/bentoml/bentoml/grpc/v1 v0.0.0-unpublished => ./github.com/bentoml/bentoml/grpc/v1

            By using `replace directive <https://go.dev/ref/mod#go-mod-file-replace>`_, we
            ensure that Go will know where our generated stubs to be imported from. (since we don't host the generate gRPC stubs on `pkg.go.dev` 😄)

            .. include:: ./snippets/grpc/additional_setup.rst

            Here is the ``protoc`` command to generate the gRPC Go stubs:

            .. code-block:: bash

               » protoc -I. -I thirdparty/protobuf/src  \
                        --go_out=. --go_opt=paths=import \
                        --go-grpc_out=. --go-grpc_opt=paths=import \
                        bentoml/grpc/v1/service.proto

            Then run the following to make sure the generated stubs are importable:

            .. code-block:: bash

               » pushd github.com/bentoml/bentoml/grpc/v1
               » go mod init v1 && go mod tidy
               » popd

      Create a ``client.go`` file with the following content:

      .. literalinclude:: ../../../grpc-client/go/client.go
         :language: go
         :caption: `client.go`

   .. tab-item:: C++
      :sync: cpp

      :bdg-info:`Requirements:` Make sure follow the `instructions <https://grpc.io/docs/languages/cpp/quickstart/#install-grpc>`_ to install gRPC and Protobuf locally.

      We will create our C++ client in the directory ``~/workspace/iris_cc_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_cc_client
         » cd ~/workspace/iris_cc_client

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            Define a |workspace|_ file:

            .. dropdown:: ``WORKSPACE``
               :icon: code

               .. literalinclude:: ./snippets/grpc/cpp/WORKSPACE.snippet.bzl
                  :language: python

            Followed by defining a |build|_ file:

            .. dropdown:: ``BUILD``
               :icon: code

               .. literalinclude:: ./snippets/grpc/cpp/BUILD.snippet.bzl
                  :language: python

         .. tab-item:: Using protoc and language-specific plugins
            :sync: protoc-and-plugins

            .. include:: ./snippets/grpc/additional_setup.rst

            Here is the ``protoc`` command to generate the gRPC C++ stubs:

            .. code-block:: bash

               » protoc -I . -I ./thirdparty/protobuf/src \
                        --cpp_out=. --grpc_out=. \
                        --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
                        bentoml/grpc/v1/service.proto

      Create a ``client.cpp`` file with the following content:

      .. literalinclude:: ../../../grpc-client/cpp/client.cc
         :language: cpp
         :caption: `client.cpp`

   .. tab-item:: Java
      :sync: java

      :bdg-info:`Requirements:` Make sure to have `JDK>=7 <https://jdk.java.net/>`_.

      :bdg-info:`Optional:`  follow the :github:`instructions <grpc/grpc-java/tree/master/compiler>` to install ``protoc`` plugin for gRPC Java if you plan to use ``protoc`` standalone.

      .. note::

         Feel free to use any Java build tools of choice (Maven, Gradle, Bazel, etc.) to build and run the client you find fit.

         In this tutorial we will be using |bazel|_.

      We will create our Java client in the directory ``~/workspace/iris_java_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_java_client
         » cd ~/workspace/iris_java_client

      Create the client Java package (``com.client.BentoServiceClient``):

      .. code-block:: bash

         » mkdir -p src/main/java/com/client

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            Define a |workspace|_ file:

            .. dropdown:: ``WORKSPACE``
               :icon: code

               .. literalinclude:: ./snippets/grpc/java/WORKSPACE.snippet.bzl
                  :language: python

            Followed by defining a |build|_ file:

            .. dropdown:: ``BUILD``
               :icon: code

               .. literalinclude:: ./snippets/grpc/java/BUILD.snippet.bzl
                  :language: python

         .. tab-item:: Using others build system
            :sync: protoc-and-plugins

            One simply can't manually running ``javac`` to compile the Java class, since
            there are way too many dependencies to be resolved.

            Provided below is an example of how one can use `gradle <https://gradle.org/>`_ to build the Java client.

            .. code-block:: bash

               » gradle init --project-dir .

            The following ``build.gradle`` should be able to help you get started:

            .. literalinclude:: ../../../grpc-client/java/build.gradle
               :language: text
               :caption: build.gradle

            To build the client, run:

            .. code-block:: bash

               » ./gradlew build

      Proceed to create a ``src/main/java/com/client/BentoServiceClient.java`` file with the following content:

      .. literalinclude:: ../../../grpc-client/java/src/main/java/com/client/BentoServiceClient.java
         :language: java
         :caption: `BentoServiceClient.java`

      .. dropdown:: On running ``protoc`` standalone (optional)
         :icon: book

         .. include:: ./snippets/grpc/additional_setup.rst

         Here is the ``protoc`` command to generate the gRPC Java stubs if you need to use ``protoc`` standalone:

         .. code-block:: bash

            » protoc -I . \
                     -I ./thirdparty/protobuf/src \
                     --java_out=./src/main/java \
                     --grpc-java_out=./src/main/java \
                     bentoml/grpc/v1/service.proto

   .. tab-item:: Kotlin
      :sync: kotlin

      :bdg-info:`Requirements:` Make sure to have the `prequisites <https://grpc.io/docs/languages/kotlin/quickstart/#prerequisites>`_ to get started with :github:`grpc/grpc-kotlin`.

      :bdg-info:`Optional:` feel free to install :github:`Kotlin gRPC codegen <grpc/grpc-kotlin/blob/master/compiler/README.md>` in order to generate gRPC stubs if you plan to use ``protoc`` standalone.

      To bootstrap the Kotlin client, feel free to use either `gradle <https://gradle.org/>`_ or
      `maven <https://maven.apache.org/>`_ to build and run the following client code.

      In this example, we will use |bazel|_ to build and run the client.

      We will create our Kotlin client in the directory ``~/workspace/iris_kotlin_client/``, followed by creating the client directory structure:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_kotlin_client
         » cd ~/workspace/iris_kotlin_client
         » mkdir -p src/main/kotlin/com/client

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            Define a |workspace|_ file:

            .. dropdown:: ``WORKSPACE``

               .. literalinclude:: ./snippets/grpc/kotlin/WORKSPACE.snippet.bzl
                  :language: python

            Followed by defining a |build|_ file:

            .. dropdown:: ``BUILD``

               .. literalinclude:: ./snippets/grpc/kotlin/BUILD.snippet.bzl
                  :language: python

         .. tab-item:: Using others build system
            :sync: protoc-and-plugins

            One simply can't manually compile all the Kotlin files, since there are way too many dependencies to be resolved.

            Provided below is an example of how one can use `gradle <https://gradle.org/>`_ to build the Kotlin client.

            .. code-block:: bash

               » gradle init --project-dir .

            The following ``build.gradle.kts`` should be able to help you get started:

            .. literalinclude:: ../../../grpc-client/kotlin/build.gradle.kts
               :language: text
               :caption: build.gradle.kts

            To build the client, run:

            .. code-block:: bash

               » ./gradlew build

      Proceed to create a ``src/main/kotlin/com/client/BentoServiceClient.kt`` file with the following content:

      .. literalinclude:: ../../../grpc-client/kotlin/src/main/kotlin/com/client/BentoServiceClient.kt
         :language: java
         :caption: `BentoServiceClient.kt`

      .. dropdown:: On running ``protoc`` standalone (optional)
         :icon: book

         .. include:: ./snippets/grpc/additional_setup.rst

         Here is the ``protoc`` command to generate the gRPC Kotlin stubs if you need to use ``protoc`` standalone:

         .. code-block:: bash

            » protoc -I. -I ./thirdparty/protobuf/src \
                     --kotlin_out ./kotlin/src/main/kotlin/ \
                     --grpc-kotlin_out ./kotlin/src/main/kotlin \
                     --plugin=protoc-gen-grpc-kotlin=$(which protoc-gen-grpc-kotlin) \
                     bentoml/grpc/v1/service.proto

   .. tab-item:: Node.js
      :sync: nodejs

      :bdg-info:`Requirements:` Make sure to have `Node.js <https://nodejs.org/en/>`_
      installed in your system.

      We will create our Node.js client in the directory ``~/workspace/iris_node_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_node_client
         » cd ~/workspace/iris_node_client

      .. dropdown:: Initialize the project and use the following ``package.json``:

         .. literalinclude:: ../../../grpc-client/node/package.json
            :language: json
            :caption: `package.json`

      Install the dependencies with either ``npm`` or ``yarn``:

      .. code-block:: bash

         » yarn install --add-devs

      .. note::

         If you are using M1, you might also have to prepend ``npm_config_target_arch=x64`` to ``yarn`` command:

         .. code-block:: bash

            » npm_config_target_arch=x64 yarn install --add-devs

      .. include:: ./snippets/grpc/additional_setup.rst

      Here is the ``protoc`` command to generate the gRPC Javascript stubs:

      .. code-block:: bash

         » $(npm bin)/grpc_tools_node_protoc \
                  -I . -I ./thirdparty/protobuf/src \
                  --js_out=import_style=commonjs,binary:. \
                  --grpc_out=grpc_js:js \
                  bentoml/grpc/v1/service.proto

      Proceed to create a ``client.js`` file with the following content:

      .. literalinclude:: ../../../grpc-client/node/client.js
         :language: javascript
         :caption: `client.js`

   .. tab-item:: Swift
      :sync: swift

      :bdg-info:`Requirements:` Make sure to have the :github:`prequisites <grpc/grpc-swift/blob/main/docs/quick-start.md#prerequisites>` to get started with :github:`grpc/grpc-swift`.

      We will create our Swift client in the directory ``~/workspace/iris_swift_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_swift_client
         » cd ~/workspace/iris_swift_client

      We will use `Swift Package Manager <https://swift.org/package-manager/>`_ to build and run the client.

      .. code-block:: bash

         » swift package init --type executable

      .. dropdown:: Initialize the project and use the following ``Package.swift``:

         .. literalinclude:: ../../../grpc-client/swift/Package.swift
            :language: swift
            :caption: `Package.swift`

      .. include:: ./snippets/grpc/additional_setup.rst

      Here is the ``protoc`` command to generate the gRPC Swift stubs:

      .. code-block:: bash

         » protoc -I. -I ./thirdparty/protobuf/src \
                  --swift_out=Sources --swift_opt=Visibility=Public \
                  --grpc-swift_out=Sources --grpc-swift_opt=Visibility=Public \
                  --plugin=protoc-gen-grpc-swift=$(which protoc-gen-grpc-swift) \
                  bentoml/grpc/v1/service.proto

      Proceed to create a ``Sources/BentoServiceClient/main.swift`` file with the following content:

      .. literalinclude:: ../../../grpc-client/swift/Sources/BentoServiceClient/main.swift
         :language: swift
         :caption: `main.swift`

   .. tab-item:: PHP
      :sync: php

      :bdg-info:`Requirements:` Make sure to follow the :github:`instructions <grpc/grpc/blob/master/src/php/README.md>` to install ``grpc`` via either `pecl <https://pecl.php.net/>`_ or from source.

      .. note::

         You will also have to symlink the built C++ extension to the PHP extension directory for it to be loaded by PHP.

      We will then use |bazel|_, `composer <https://getcomposer.org/>`_ to build and run the client.

      We will create our PHP client in the directory ``~/workspace/iris_php_client/``:

      .. code-block:: bash

         » mkdir -p ~/workspace/iris_php_client
         » cd ~/workspace/iris_php_client

      Create a new PHP package:

      .. code-block:: bash

         » composer init

      .. dropdown:: An example ``composer.json`` for the client:
         :icon: code

         .. literalinclude:: ../../../grpc-client/php/composer.json
            :language: json

      .. include:: ./snippets/grpc/additional_setup.rst

      Here is the ``protoc`` command to generate the gRPC swift stubs:

      .. code-block:: bash

         » protoc -I . -I ./thirdparty/protobuf/src \
                  --php_out=. \
                  --grpc_out=. \
                  --plugin=protoc-gen-grpc=$(which grpc_php_plugin) \
                  bentoml/grpc/v1/service.proto

      Proceed to create a ``BentoServiceClient.php`` file with the following content:

      .. literalinclude:: ../../../grpc-client/php/BentoServiceClient.php
         :language: php
         :caption: `BentoServiceClient.php`

.. TODO::

   Bazel instruction for ``swift``, ``nodejs``, ``python``

:raw-html:`<br />`

Then you can proceed to run the client scripts:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: bash

         » python -m client

   .. tab-item:: Go
      :sync: golang

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            .. code-block:: bash

               » bazel run //:client_go

         .. tab-item:: Using protoc and language-specific plugins
            :sync: protoc-and-plugins

            .. code-block:: bash

               » go run ./client.go

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            .. code-block:: bash

               » bazel run :client_cc

         .. tab-item:: Using protoc and language-specific plugins
            :sync: protoc-and-plugins

            Refer to :github:`grpc/grpc` for instructions on using CMake and other similar build tools.

      .. note::

         See the :github:`instructions on GitHub <bentoml/BentoML/tree/main/grpc-client/README.md>` for working C++ client.

   .. tab-item:: Java
      :sync: java

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            .. code-block:: bash

               » bazel run :client_java

         .. tab-item:: Using others build system
            :sync: protoc-and-plugins

            We will use ``gradlew`` to build the client and run it:

            .. code-block:: bash

               » ./gradlew build && \
                  ./build/tmp/scripts/bentoServiceClient/bento-service-client

      .. note::

         See the :github:`instructions on GitHub <bentoml/BentoML/tree/main/grpc-client/README.md>` for working Java client.

   .. tab-item:: Kotlin
      :sync: kotlin

      .. tab-set::

         .. tab-item:: Using bazel (recommended)
            :sync: bazel-workflow

            .. code-block:: bash

               » bazel run :client_kt

         .. tab-item:: Using others build system
            :sync: protoc-and-plugins

            We will use ``gradlew`` to build the client and run it:

            .. code-block:: bash

               » ./gradlew build && \
                  ./build/tmp/scripts/bentoServiceClient/bento-service-client

      .. note::

         See the :github:`instructions on GitHub <bentoml/BentoML/tree/main/grpc-client/README.md>` for working Kotlin client.

   .. tab-item:: Node.js
      :sync: nodejs

      .. code-block:: bash

         » node client.js

   .. tab-item:: Swift
      :sync: swift

      .. code-block:: bash

         » swift run BentoServiceClient

   .. tab-item:: PHP
      :sync: php

      .. code-block:: bash

         » php -d extension=/path/to/grpc.so -d max_execution_time=300 BentoServiceClient.php


.. dropdown:: Additional language support for client implementation
   :icon: triangle-down

   .. tab-set::

      .. tab-item:: Ruby
         :sync: ruby

         :bdg-primary:`Note:` Please check out the :github:`gRPC Ruby <grpc/grpc/blob/master/src/ruby/README.md#grpc-ruby>` for how to install from source.
         Check out the :github:`examples folder <grpc/grpc/blob/master/examples/ruby/README.md#prerequisites>` for Ruby client implementation.

      .. tab-item:: .NET
         :sync: dotnet

         :bdg-primary:`Note:` Please check out the :github:`gRPC .NET <grpc/grpc-dotnet/tree/master/examples>` examples folder for :github:`grpc/grpc-dotnet` client implementation.

      .. tab-item:: Dart
         :sync: dart

         :bdg-primary:`Note:` Please check out the :github:`gRPC Dart <grpc/grpc-dart/tree/master/examples>` examples folder for :github:`grpc/grpc-dart` client implementation.

      .. tab-item:: Rust
         :sync: rust

         :bdg-primary:`Note:` Currently there are no official gRPC Rust client implementation. Please check out the :github:`tikv/grpc-rs` as one of the unofficial implementation.


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

``--enable-features`` allows users to containerize any of the existing Bentos with :ref:`additional features <concepts/bento:Enable features for your Bento>` that BentoML provides without having to rebuild the Bento.

.. note::

   ``--enable-features`` accepts a comma-separated list of features or multiple arguments.

After containerization, your Bento container can now be used with gRPC:

.. code-block:: bash

   » docker run -it --rm \
                -p 3000:3000 -p 3001:3001 \
                iris_classifier:6otbsmxzq6lwbgxi serve-grpc

Congratulations! You have successfully served, containerized and tested your BentoService with gRPC.

-------------

Using gRPC in BentoML
---------------------

We will dive into some of the details of how gRPC is implemented in BentoML.

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

      .. tab-item:: v1

         .. literalinclude:: ../../../src/bentoml/grpc/v1/service.proto
            :language: protobuf

      .. tab-item:: v1alpha1

         .. literalinclude:: ../../../src/bentoml/grpc/v1alpha1/service.proto
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
| :ref:`guides/grpc:Series representation via ``Series```          | :ref:`bentoml.io.PandasDataFrame <reference/api_io_descriptors:Tabular Data with Pandas>` |
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

      .. literalinclude:: ./snippets/grpc/python/request.py
         :language: python

   .. tab-item:: Go
      :sync: golang

      .. literalinclude:: ./snippets/grpc/go/request.go
         :language: go

   .. tab-item:: C++
      :sync: cpp

      .. literalinclude:: ./snippets/grpc/cpp/request.cc
         :language: cpp

   .. tab-item:: Java
      :sync: java

      .. literalinclude:: ./snippets/grpc/java/Request.java
         :language: java

   .. tab-item:: Kotlin
      :sync: kotlin

      .. literalinclude:: ./snippets/grpc/kotlin/Request.kt
         :language: java

   .. tab-item:: Node.js
      :sync: nodejs

      .. literalinclude:: ./snippets/grpc/node/request.js
         :language: javascript

   .. tab-item:: Swift
      :sync: swift

      .. literalinclude:: ./snippets/grpc/swift/Request.swift
         :language: swift



Array representation via ``NDArray``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:bdg-info:`Description:` ``NDArray`` represents a flattened n-dimensional array of arbitrary type. It accepts the following fields:

* `dtype`

  The data type of given input. This is a `Enum <https://developers.google.com/protocol-buffers/docs/proto3#enum>`_ field that provides 1-1 mapping with Protobuf data types to NumPy data types:

  +-----------------------+---------------+------------+
  | pb.NDArray.DType      | numpy.dtype   | Enum value |
  +=======================+===============+============+
  | ``DTYPE_UNSPECIFIED`` | ``None``      | 0          |
  +-----------------------+---------------+------------+
  | ``DTYPE_FLOAT``       | ``np.float``  | 1          |
  +-----------------------+---------------+------------+
  | ``DTYPE_DOUBLE``      | ``np.double`` | 2          |
  +-----------------------+---------------+------------+
  | ``DTYPE_BOOL``        | ``np.bool_``  | 3          |
  +-----------------------+---------------+------------+
  | ``DTYPE_INT32``       | ``np.int32``  | 4          |
  +-----------------------+---------------+------------+
  | ``DTYPE_INT64``       | ``np.int64``  | 5          |
  +-----------------------+---------------+------------+
  | ``DTYPE_UINT32``      | ``np.uint32`` | 6          |
  +-----------------------+---------------+------------+
  | ``DTYPE_UINT64``      | ``np.uint64`` | 7          |
  +-----------------------+---------------+------------+
  | ``DTYPE_STRING``      | ``np.str_``   | 8          |
  +-----------------------+---------------+------------+

* `shape`

  A list of `int32` that represents the shape of the flattened array. the :ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>` will
  then reshape the given payload into expected shape.

  Note that this value will always takes precendence over the ``shape`` field in the :ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>` descriptor,
  meaning the array will be reshaped to this value first if given. Refer to :meth:`bentoml.io.NumpyNdarray.from_proto` for implementation details.

* `string_values`, `float_values`, `double_values`, `bool_values`, `int32_values`, `int64_values`, `uint32_values`, `unit64_values`

  Each of the fields is a `list` of the corresponding data type. The list is a flattened array, and will be reconstructed
  alongside with ``shape`` field to the original payload.

  Per request sent, one message should only contain **ONE** of the aforementioned fields.

  The interaction among the above fields and ``dtype`` are as follows:

  - if ``dtype`` is not present in the message:
      * All of the fields are empty, then we return a ``np.empty``.
      * We will loop through all of the provided fields, and only allows one field per message.

        If here are more than one field (i.e. ``string_values`` and ``float_values``), then we will raise an error, as we don't know how to deserialize the data.

  - otherwise:
      * We will use the provided dtype-to-field map to get the data from the given message.

      +------------------+-------------------+
      | DType            | field             |
      +------------------+-------------------+
      | ``DTYPE_BOOL``   | ``bool_values``   |
      +------------------+-------------------+
      | ``DTYPE_DOUBLE`` | ``double_values`` |
      +------------------+-------------------+
      | ``DTYPE_FLOAT``  | ``float_values``  |
      +------------------+-------------------+
      | ``DTYPE_INT32``  | ``int32_values``  |
      +------------------+-------------------+
      | ``DTYPE_INT64``  | ``int64_values``  |
      +------------------+-------------------+
      | ``DTYPE_STRING`` | ``string_values`` |
      +------------------+-------------------+
      | ``DTYPE_UINT32`` | ``uint32_values`` |
      +------------------+-------------------+
      | ``DTYPE_UINT64`` | ``uint64_values`` |
      +------------------+-------------------+

  For example, if ``dtype`` is ``DTYPE_FLOAT``, then the payload expects to have ``float_values`` field.

.. grid:: 2

    .. grid-item-card::  ``Python API``

      .. code-block:: python

         NumpyNdarray.from_sample(
            np.array([[5.4, 3.4, 1.5, 0.4]])
         )

    .. grid-item-card::  ``pb.NDArray``

      .. code-block:: none

         ndarray {
           dtype: DTYPE_FLOAT
           shape: 1
           shape: 4
           float_values: 5.4
           float_values: 3.4
           float_values: 1.5
           float_values: 0.4
         }


:bdg-primary:`API reference:` :meth:`bentoml.io.NumpyNdarray.from_proto`

:raw-html:`<br />`

Tabular data representation via ``DataFrame``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:bdg-info:`Description:` ``DataFrame`` represents any tabular data type. Currently we only support the columns orientation
since it is best for preserving the input order.

It accepts the following fields:

* `column_names`

  A list of `string` that represents the column names of the given tabular data.

* `column_values`

  A list of `Series` where `Series` represents a series of arbitrary data type. The allowed fields for
  `Series` as similar to the ones in `NDArray`:

  * one of [`string_values`, `float_values`, `double_values`, `bool_values`, `int32_values`, `int64_values`, `uint32_values`, `unit64_values`]

.. grid:: 2

    .. grid-item-card::  ``Python API``

      .. code-block:: python

         PandasDataFrame.from_sample(
             pd.DataFrame({
               "age": [3, 29],
               "height": [94, 170],
               "weight": [31, 115]
             }),
             orient="columns",
         )

    .. grid-item-card::  ``pb.DataFrame``

      .. code-block:: none

         dataframe {
           column_names: "age"
           column_names: "height"
           column_names: "weight"
           columns {
             int32_values: 3
             int32_values: 29
           }
           columns {
             int32_values: 40
             int32_values: 190
           }
           columns {
             int32_values: 140
             int32_values: 178
           }
         }

:bdg-primary:`API reference:` :meth:`bentoml.io.PandasDataFrame.from_proto`

Series representation via ``Series``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:bdg-info:`Description:` ``Series`` portrays a series of values. This can be used for representing Series types in tabular data.

It accepts the following fields:

* `string_values`, `float_values`, `double_values`, `bool_values`, `int32_values`, `int64_values`

  Similar to NumpyNdarray, each of the fields is a `list` of the corresponding data type. The list is a 1-D array, and will be then pass to ``pd.Series``.

  Each request should only contain **ONE** of the aforementioned fields.

  The interaction among the above fields and ``dtype`` from ``PandasSeries`` are as follows:

  - if ``dtype`` is not present in the descriptor:
      * All of the fields are empty, then we return an empty ``pd.Series``.
      * We will loop through all of the provided fields, and only allows one field per message.

        If here are more than one field (i.e. ``string_values`` and ``float_values``), then we will raise an error, as we don't know how to deserialize the data.

  - otherwise:
      * We will use the provided dtype-to-field map to get the data from the given message.

.. grid:: 2

    .. grid-item-card::  ``Python API``

      .. code-block:: python

         PandasSeries.from_sample([5.4, 3.4, 1.5, 0.4])

    .. grid-item-card::  ``pb.Series``

      .. code-block:: none

         series {
           float_values: 5.4
           float_values: 3.4
           float_values: 1.5
           float_values: 0.4
         }


:bdg-primary:`API reference:` :meth:`bentoml.io.PandasSeries.from_proto`

:raw-html:`<br />`

File-like object via ``File``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:bdg-info:`Description:` ``File`` represents any arbitrary file type. this can be used
to send in any file type, including images, videos, audio, etc.

.. note::

   Currently both :class:`bentoml.io.File` and :class:`bentoml.io.Image` are using
   ``pb.File``

It accepts the following fields:

* `content`

  A `bytes` field that represents the content of the file.

* `kind`

  An optional `string` field that represents the file type. If specified, it will raise an error if
  ``mime_type`` specified in :ref:`bentoml.io.File <reference/api_io_descriptors:Files>` is not matched.

.. grid:: 2

    .. grid-item-card::  ``Python API``

      .. code-block:: python

         Image(mime_type="application/pdf")

    .. grid-item-card::  ``pb.File``

      .. code-block:: none

         file {
           kind: "application/pdf"
           content: <bytes>
         }


:ref:`bentoml.io.Image <reference/api_io_descriptors:Images>` will also be using ``pb.File``.

.. grid:: 2

    .. grid-item-card::  ``Python API``

      .. code-block:: python

         File(mime_type="image/png")

    .. grid-item-card::  ``pb.File``

      .. code-block:: none

         file {
           kind: "image/png"
           content: <bytes>
         }


Complex payload via ``Multipart``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:bdg-info:`Description:` ``Multipart`` represents a complex payload that can contain
multiple different fields. It takes a ``fields``, which is a dictionary of input name to
its coresponding :class:`bentoml.io.IODescriptor`

.. grid:: 2

    .. grid-item-card::  ``Python API``

      .. code-block:: python

         Multipart(
            meta=Text(),
            arr=NumpyNdarray(
               dtype=np.float16,
               shape=[2,2]
            )
         )

    .. grid-item-card::  ``pb.Multipart``

      .. code-block:: none

         multipart {
            fields {
               key: "arr"
               value {
                  ndarray {
                  dtype: DTYPE_FLOAT
                  shape: 2
                  shape: 2
                  float_values: 1.0
                  float_values: 2.0
                  float_values: 3.0
                  float_values: 4.0
                  }
               }
            }
            fields {
               key: "meta"
               value {
                  text {
                  value: "nlp"
                  }
               }
            }
         }

:bdg-primary:`API reference:` :meth:`bentoml.io.Multipart.from_proto`

Compact data format via ``serialized_bytes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``serialized_bytes`` field in both ``Request`` and ``Response``  is reserved for pre-established protocol encoding between client and server.

BentoML leverages the field to improve serialization performance between BentoML client and server. Thus the field is not **recommended** for use directly.

Mounting Servicer
~~~~~~~~~~~~~~~~~

gRPC service :ref:`multiplexing <guides/grpc:Demystifying the misconception of gRPC vs. REST>` enables us to mount additional custom servicers alongside with BentoService,
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

.. image:: /_static/img/interceptor-flow.png
   :alt: Interceptor Flow

Since interceptors are executed in the order they are added, users interceptors will be executed after the built-in interceptors.

   Users interceptors shouldn't modify the existing headers and data of the incoming ``Request``.

BentoML currently only support **async interceptors** (via `grpc.aio.ServerInterceptor <https://grpc.github.io/grpc/python/grpc_asyncio.html#grpc.aio.ServerInterceptor>`_, as opposed to `grpc.ServerInterceptor <https://grpc.github.io/grpc/python/grpc_asyncio.html#grpc.aio.ServerInterceptor>`_). This is
because BentoML gRPC server is an async implementation of gRPC server.

.. note::

   If you are using ``grpc.ServerInterceptor``, you will need to migrate it over
   to use the new ``grpc.aio.ServerInterceptor`` in order to use this feature.

   Feel free to reach out to us at `#support on Slack <https://l.bentoml.com/join-slack>`_

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

---------------

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

* **Lack of support for additional content types**: gRPC depends on protobuf, its content
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

BentoML allows user to tune the performance of gRPC via :ref:`bentoml_configuration.yaml <guides/configuration:Configuration>` via ``api_server.grpc``.

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

.. epigraph::
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

.. epigraph::
   :bdg-info:`Definition:` The maximum number of concurrent RPCs this server will service before returning ``RESOURCE_EXHAUSTED`` status.

By default we set to ``None`` to indicate no limit, and let gRPC to decide the limit.

:raw-html:`<br />`

``max_message_length``
^^^^^^^^^^^^^^^^^^^^^^

.. epigraph::
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
