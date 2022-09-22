=================
Serving with gRPC
=================

This guide will demonstrate advanced features that BentoML offers for you to get started
with `gRPC <https://grpc.io/>`_:

- First-class support for :ref:`custom gRPC Servicer <guides/grpc:Mounting Servicer>`, :ref:`custom interceptors <guides/grpc:Mounting gRPC Interceptors>`, handlers.
- Adding gRPC support to existing Bento.

This guide will also walk your through some of the strengths and weaknesses of serving with gRPC, as well as
recommendation on scenarios where gRPC might be a good fit. We will be using the example service from :ref:`the quickstart<tutorial:Tutorial: Intro to BentoML>` to interact and explore said gRPC features.

:bdg-info:`Requirements:` This guide assumes that you have basic knowledge of gRPC and protobuf. If you aren't
familar with gRPC, you can start with gRPC `quick start guide <https://grpc.io/docs/languages/python/quickstart/>`_.

.. note::

   For quick introduction to serving with gRPC, see :ref:`Intro to BentoML <tutorial:Tutorial: Intro to BentoML>`

Why you may need this?
----------------------

- If gRPC is the required :wiki:`RPC <Remote_procedure_call>` framework your
  organization's business logics.
- If organization codebase exists in a polygot environment, and you want to communicate your ML application
  with services implemented in different languages.
- If you are simply curious about BentoML's support for gRPC 😊.

Get started with gRPC in BentoML
--------------------------------

Requirements
~~~~~~~~~~~~

Install BentoML with gRPC support with :pypi:`pip`:

.. code-block:: bash

   » pip install "bentoml[grpc]"

Thats it! You can now serving your Bento with gRPC via :ref:`bentoml serve-grpc <reference/cli:serve-grpc>` without having to modify your current service definition 😃.

.. code-block:: bash

   » bentoml serve-grpc iris_classifier:latest --production

Client implementation
~~~~~~~~~~~~~~~~~~~~~

From another terminal, use one of the following client implementation to send request to the
gRPC server:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python
         :caption: `client.py`

         if __name__ == "__main__":
            import asyncio

            import grpc

            from bentoml.grpc.utils import import_generated_stubs

            pb, services = import_generated_stubs()
            async def run():
               async with grpc.aio.insecure_channel("localhost:3000") as channel:
                     stub = services.BentoServiceStub(channel)
                     req = stub.Call(
                        request=pb.Request(
                           api_name="predict",
                           ndarray=pb.NDArray(
                                 dtype=pb.NDArray.DTYPE_FLOAT,
                                 shape=(1, 4),
                                 float_values=[5.9, 3, 5.1, 1.8],
                           ),
                        )
                     )
               print(req)

            asyncio.run(run())

   .. tab-item:: Go
      :sync: golang

      .. code-block:: go
         :caption: `client.go`

         package client

         import (
            "context"
            "fmt"
            "time"

            pb "bentoml/grpc/v1alpha1"

            "google.golang.org/grpc"
            "google.golang.org/grpc/credentials/insecure"
         )

         var opts []grpc.DialOption

         const serverAddr = "localhost:3000"

         func main() {
            opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
            conn, err := grpc.Dial(serverAddr, opts...)
            if err != nil {
               panic(err)
            }
            defer conn.Close()
            ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
            defer cancel()

            client := pb.NewBentoServiceClient(conn)

            resp, err := client.Call(ctx, &pb.Request{ApiName: "predict", Content: &pb.Request_Ndarray{Ndarray: &pb.NDArray{Dtype: *pb.NDArray_DTYPE_FLOAT.Enum(), Shape: []int32{1, 4}, FloatValues: []float32{3.5, 2.4, 7.8, 5.1}}}})
            if err != nil {
               panic(err)
            }
            fmt.Print(resp)
         }

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp
         :caption: `client.cpp`

         #include <array>
         #include <iostream>
         #include <memory>
         #include <mutex>
         #include <string>
         #include <vector>

         #include <grpc/grpc.h>
         #include <grpcpp/channel.h>
         #include <grpcpp/client_context.h>
         #include <grpcpp/create_channel.h>
         #include <grpcpp/grpcpp.h>
         #include <grpcpp/security/credentials.h>

         #include "bentoml/grpc/v1alpha1/service.grpc.pb.h"
         #include "bentoml/grpc/v1alpha1/service.pb.h"

         using bentoml::grpc::v1alpha1::BentoService;
         using bentoml::grpc::v1alpha1::NDArray;
         using bentoml::grpc::v1alpha1::Request;
         using bentoml::grpc::v1alpha1::Response;
         using grpc::Channel;
         using grpc::ClientAsyncResponseReader;
         using grpc::ClientContext;
         using grpc::CompletionQueue;
         using grpc::Status;

         int main(int argc, char **argv) {
             auto stubs = BentoService::NewStub(grpc::CreateChannel(
                   "localhost:3000", grpc::InsecureChannelCredentials()));
             std::vector<float> data = {3.5, 2.4, 7.8, 5.1};
             std::vector<int> shape = {1, 4};

             Request request;
             request.set_api_name("predict");

             NDArray *ndarray = request.mutable_ndarray();
             ndarray->mutable_shape()->Assign(shape.begin(), shape.end());
             ndarray->mutable_float_values()->Assign(data.begin(), data.end());

             Response resp;
             ClientContext context;

             // Storage for the status of the RPC upon completion.
             Status status = stubs->Call(&context, request, &resp);

             // Act upon the status of the actual RPC.
             if (!status.ok()) {
                std::cout << status.error_code() << ": " << status.error_message()
                         << std::endl;
                return 1;
             }
             if (!resp.has_ndarray()) {
                std::cout << "Currently only accept output as NDArray." << std::endl;
                return 1;
             }
             std::cout << "response byte size: " << resp.ndarray().ByteSizeLong()
                         << std::endl;
         }


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

      To compile C++ client, we need to somehow include the protobuf and gRPC C++
      headers and use either clangd, g++ or `bazel <https://bazel.build/>`_ to compile
      the binary.

      Since this is outside of the scope of this guide, we will leave the details on how to
      compile the C++ client to the reader. Below is a gist of how one can use
      Bazel to compile the C++ client for those who are interested:

      .. dropdown:: Bazel instruction

         After installing bazel, define a ``WORKSPACE`` file in the same directory as
         ``client.cpp``:

         .. dropdown:: ``WORKSPACE``

            .. code-block:: python

               workspace(name = "client")

               load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

               http_archive(
                  name = "rules_proto",
                  sha256 = "e017528fd1c91c5a33f15493e3a398181a9e821a804eb7ff5acdd1d2d6c2b18d",
                  strip_prefix = "rules_proto-4.0.0-3.20.0",
                  urls = [
                     "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0-3.20.0.tar.gz",
                  ],
               )
               http_archive(
                  name = "rules_proto_grpc",
                  sha256 = "507e38c8d95c7efa4f3b1c0595a8e8f139c885cb41a76cab7e20e4e67ae87731",
                  strip_prefix = "rules_proto_grpc-4.1.1",
                  urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.1.1.tar.gz"],
               )

               load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
               load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")

               rules_proto_grpc_toolchains()
               rules_proto_grpc_repos()
               rules_proto_dependencies()
               rules_proto_toolchains()

         Then follow by defining a ``BUILD`` file:

         .. dropdown:: ``BUILD``

            .. code-block:: python

               load("@rules_proto//proto:defs.bzl", "proto_library")
               load("@rules_proto_grpc//cpp:defs.bzl", "cc_grpc_library", "cc_proto_library")

               proto_library(
                  name = "service_proto",
                  srcs = ["bentoml/grpc/v1alpha1/service.proto"],
                  deps = ["@com_google_protobuf//:struct_proto", "@com_google_protobuf//:wrappers_proto"]
               )

               cc_proto_library(
                  name = "service_cc",
                  protos = [":service_proto"],
               )

               cc_grpc_library(
                  name = "service_cc_grpc",
                  protos = [":service_proto"],
                  deps = [":service_cc"],
               )

               cc_binary(
                  name = "client_cc",
                  srcs = ["client.cc"],
                  deps = [
                     ":service_cc_grpc",
                     "@com_github_grpc_grpc//:grpc++",
                  ],
               )

         Proceed then to run the client:

         .. code-block:: bash

            » bazel run :client_cc

After successfully running the client, proceed to build the bento as usual:

.. code-block:: bash

   » bentoml build


To containerize the Bento with gRPC features, pass in ``--enable-features=grpc`` to
:ref:`bentoml containerize <reference/cli:containerize-enable-features>` to add additional gRPC
dependencies to your Bento

.. code-block:: bash

   » bentoml containerize iris_classifier:latest --enable-features=grpc

``--enable-features`` allows users to containerize any of the existing Bentos with :ref:`additional features </installation:Additional features>` without having to rebuild the Bento.

.. note::

   ``--enable-features`` accepts a comma-separated list of features or multiple arguments.

After containerization, your Bento container can now be used with gRPC:

.. code-block:: bash

   » docker run -it --rm -p 3000:3000 -p 3001:3001 iris_classifier:6otbsmxzq6lwbgxi serve-grpc --production

Use one of the above :ref:`client implementation <guides/grpc:Client implementation>` to
send test requests to your containerized BentoService.

Congratulations! You have successfully served, containerized and tested your BentoService with gRPC.

Interact with gPRC
------------------

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

* ``api_name``: the name of the API function defined inside your BentoService. 
* `oneof <https://developers.google.com/protocol-buffers/docs/proto3#oneof>`_ ``content``: the field can be one of the following types:

   * ``NDArray``
   * ``DataFrame``
   * ``Series``
   * ``File``
   * |google_protobuf_string_value|_
   * |google_protobuf_value|_
   * ``Multipart``
   * ``bytes``

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
            api_name="predict",
            ndarray=pb.NDArray(
               dtype=pb.NDArray.DTYPE_FLOAT, shape=(1, 4), float_values=[5.9, 3, 5.1, 1.8]
            ),
         )

   .. tab-item:: Go
      :sync: golang

      .. code-block:: go

         req := &pb.Request{
            ApiName: "predict",
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
         request.set_api_name("predict");

         NDArray *ndarray = request.mutable_ndarray();
         ndarray->mutable_shape()->Assign(shape.begin(), shape.end());
         ndarray->mutable_float_values()->Assign(data.begin(), data.end());

Mounting Servicer
-----------------

Since gRPC is designed for HTTP/2, one of the more powerful features it offers is
multiplexing of multiple HTTP/2 calls over a single TCP connection, which address the
phenomenon of :wiki:`head-of-line blocking <Head-of-line_blocking>`.

This allows us to mount multiple gRPC servicer alongside the BentoService gRPC servicer,
and serve them all under the same port.

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

.. note::

   ``service_names`` is **REQUIRED** here, as this will be used for `reflection <https://github.com/grpc/grpc/blob/master/doc/server-reflection.md>`_
   when ``--enable-reflection`` is passed to ``bentoml serve-grpc``.

Mounting gRPC Interceptors
--------------------------

Inteceptors are a component of gRPC that allows us to intercept and interact with the
proto message and service context either before - or after - the actual RPC call was
sent/received by client/server.

Interceptors to gRPC is what middleware is to HTTP. The most common use-case for Interceptors
are authentication, :ref:`tracing <guides/tracing>`, access logs, and more.

BentoML comes with a sets of built-in *async interceptors* to provide support for access logs,
`OpenTelemetry <https://opentelemetry.io/>`_, and `Prometheus <https://prometheus.io/>`_.

The following diagrams demonstrates the flow of a gRPC request from client to server:

.. image:: /_static/img/interceptor-flow.svg
   :alt: Interceptor Flow

Since Interceptors are executed in the order they are added, users interceptors will be executed after the built-in interceptors.

This also means that users interceptors should be **READ-ONLY**, and shouldn't modify the state of the
incoming request.

BentoML currently only support **async interceptors** (created using
``grpc.aio.ServerInterceptor``, as opposed to ``grpc.ServerInterceptor``). This is
because BentoML gRPC server is an async implementation of gRPC server.

.. note::

   If you are using ``grpc.ServerInterceptor``, you will need to migrate it over
   to use the new ``grpc.aio.ServerInterceptor`` in order to use this feature.

To add your intercptors to existing BentoService, use ``svc.add_grpc_interceptor``:

.. code-block:: Python
   :caption: `service.py`

   svc.add_grpc_interceptor(MyInterceptor)

.. note::

   ``add_grpc_interceptor`` also supports `partial` class as well as multiple arguments
   interceptors:

   .. tab-set::

      .. tab-item:: multiple arugments

         .. code-block:: Python

            svc.add_grpc_interceptor(MyInterceptor, arg1="foo", arg2="bar")

      .. tab-item:: partial

         .. code-block:: Python

            from functools import partial

            svc.add_grpc_interceptor(partial(MyInterceptor, arg1="foo", arg2="bar"))


Recommendation for gRPC usage
-----------------------------


