=================
Serving with gRPC
=================

This guide will demonstrate advanced features that BentoML offers for you to get started
with `gRPC <https://grpc.io/>`_:

- First-class support for custom servicer, interceptors, handlers.
- Adding gRPC support to existing BentoService container.

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

Install BentoML with gRPC support with :pypi:`pip`:

.. code-block:: bash

   » pip install "bentoml[grpc]"

Thats it! You can now serving your Bento with gRPC via :ref:`bentoml serve-grpc <reference/cli:serve-grpc>` without having to modify your current service definition 😃.

.. code-block:: bash

   » bentoml serve-grpc iris_classifier:latest --production

From another terminal, use one of the following client implementation to send request to the
gRPC server:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::

         .. tab-item:: Async
            :sync: async-api

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

         .. tab-item:: Sync
            :sync: sync-api

            .. code-block:: python
               :caption: `client.py`


               if __name__ == "__main__":
                  import grpc

                  from bentoml.grpc.utils import import_generated_stubs

                  pb, services = import_generated_stubs()
                  with grpc.insecure_channel("localhost:3000") as channel:
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

   .. tab-item:: Go
      :sync: golang

      .. code-block:: go
         :caption: client.go

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

Congratulations! You have successfully served your BentoService with gRPC.

Using the Service
-----------------

Let's take a quick look at `protobuf <https://developers.google.com/protocol-buffers/docs/overview>`_ definition of the BentoService:

.. tab-set-code::

    .. literalinclude:: ../../../bentoml/grpc/v1alpha1/service.proto
        :language: protobuf

As you can see, we define a `simple rpc` ``Call`` that sends a ``Request`` message and returns a ``Response`` message.

A ``Request`` message takes in an ``api_name`` field, which is the name of the API
function defined inside your BentoService. The ``content`` field is a `oneof <https://developers.google.com/protocol-buffers/docs/proto3#oneof>`_ field,
which means that only one fields can be set at a time. The ``content`` field can be one of the following types:

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

For example, in the :ref:`quickstart guide<tutorial:Creating a Service>`, we defined a ``classify`` API that takes in a :ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`,
which means our ``Request`` message would look like this:

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
