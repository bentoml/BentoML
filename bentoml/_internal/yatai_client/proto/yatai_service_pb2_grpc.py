# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import bentoml._internal.yatai_client.proto.deployment_pb2 as deployment__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import bentoml._internal.yatai_client.proto.repository_pb2 as repository__pb2
import bentoml._internal.yatai_client.proto.yatai_service_pb2 as yatai__service__pb2


class YataiStub(object):
    """Yatai RPC Server

    A stateful service that provides a complete BentoML model management
    and model serving/deployment workflow

    It provides two sets of APIs:
    Bento Repository: Manages saved Bento files, and making them available
    for serving in production environments
    Serving Deployment: Deploys saved Bento to a varity of different cloud
    platforms, track deploym//ent status, set up logging
    monitoring for your model serving workloads, and
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.HealthCheck = channel.unary_unary(
                '/bentoml.Yatai/HealthCheck',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=yatai__service__pb2.HealthCheckResponse.FromString,
                )
        self.GetYataiServiceVersion = channel.unary_unary(
                '/bentoml.Yatai/GetYataiServiceVersion',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=yatai__service__pb2.GetYataiServiceVersionResponse.FromString,
                )
        self.ApplyDeployment = channel.unary_unary(
                '/bentoml.Yatai/ApplyDeployment',
                request_serializer=deployment__pb2.ApplyDeploymentRequest.SerializeToString,
                response_deserializer=deployment__pb2.ApplyDeploymentResponse.FromString,
                )
        self.DeleteDeployment = channel.unary_unary(
                '/bentoml.Yatai/DeleteDeployment',
                request_serializer=deployment__pb2.DeleteDeploymentRequest.SerializeToString,
                response_deserializer=deployment__pb2.DeleteDeploymentResponse.FromString,
                )
        self.GetDeployment = channel.unary_unary(
                '/bentoml.Yatai/GetDeployment',
                request_serializer=deployment__pb2.GetDeploymentRequest.SerializeToString,
                response_deserializer=deployment__pb2.GetDeploymentResponse.FromString,
                )
        self.DescribeDeployment = channel.unary_unary(
                '/bentoml.Yatai/DescribeDeployment',
                request_serializer=deployment__pb2.DescribeDeploymentRequest.SerializeToString,
                response_deserializer=deployment__pb2.DescribeDeploymentResponse.FromString,
                )
        self.ListDeployments = channel.unary_unary(
                '/bentoml.Yatai/ListDeployments',
                request_serializer=deployment__pb2.ListDeploymentsRequest.SerializeToString,
                response_deserializer=deployment__pb2.ListDeploymentsResponse.FromString,
                )
        self.AddBento = channel.unary_unary(
                '/bentoml.Yatai/AddBento',
                request_serializer=repository__pb2.AddBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.AddBentoResponse.FromString,
                )
        self.UpdateBento = channel.unary_unary(
                '/bentoml.Yatai/UpdateBento',
                request_serializer=repository__pb2.UpdateBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.UpdateBentoResponse.FromString,
                )
        self.GetBento = channel.unary_unary(
                '/bentoml.Yatai/GetBento',
                request_serializer=repository__pb2.GetBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.GetBentoResponse.FromString,
                )
        self.DangerouslyDeleteBento = channel.unary_unary(
                '/bentoml.Yatai/DangerouslyDeleteBento',
                request_serializer=repository__pb2.DangerouslyDeleteBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.DangerouslyDeleteBentoResponse.FromString,
                )
        self.ListBento = channel.unary_unary(
                '/bentoml.Yatai/ListBento',
                request_serializer=repository__pb2.ListBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.ListBentoResponse.FromString,
                )
        self.ContainerizeBento = channel.unary_unary(
                '/bentoml.Yatai/ContainerizeBento',
                request_serializer=repository__pb2.ContainerizeBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.ContainerizeBentoResponse.FromString,
                )
        self.UploadBento = channel.stream_unary(
                '/bentoml.Yatai/UploadBento',
                request_serializer=repository__pb2.UploadBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.UploadBentoResponse.FromString,
                )
        self.DownloadBento = channel.unary_stream(
                '/bentoml.Yatai/DownloadBento',
                request_serializer=repository__pb2.DownloadBentoRequest.SerializeToString,
                response_deserializer=repository__pb2.DownloadBentoResponse.FromString,
                )


class YataiServicer(object):
    """Yatai RPC Server

    A stateful service that provides a complete BentoML model management
    and model serving/deployment workflow

    It provides two sets of APIs:
    Bento Repository: Manages saved Bento files, and making them available
    for serving in production environments
    Serving Deployment: Deploys saved Bento to a varity of different cloud
    platforms, track deploym//ent status, set up logging
    monitoring for your model serving workloads, and
    """

    def HealthCheck(self, request, context):
        """Health check ping endpoint
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetYataiServiceVersion(self, request, context):
        """Return current service version information
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ApplyDeployment(self, request, context):
        """Create new or update existing deployment
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteDeployment(self, request, context):
        """Delete existing deployment
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDeployment(self, request, context):
        """Get deployment specification (desired state)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DescribeDeployment(self, request, context):
        """Get deployment status (current state)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListDeployments(self, request, context):
        """List active deployments, by default this will return all active deployments
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AddBento(self, request, context):
        """Add new saved Bento to repository by providing the Bento name and version
        this will return an upload address that allows client to upload the bento files
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateBento(self, request, context):
        """RPC for updating a previously added Bento's information, including
        the BentoService's Metadata(apis, env, artifacts etc) and the upload status.
        Yatai server expects the client to use this RPC for notifying that, for a
        previously requested AddBento call, what's the uploading progress and when the
        upload is completed
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetBento(self, request, context):
        """Get a file path to the saved Bento files, path must be accessible form client
        machine either through HTTP, FTP, etc
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DangerouslyDeleteBento(self, request, context):
        """Deleting the Bento files that was added to this Yatai server earlier, this may
        break existing deployments or create issues when doing deployment rollback
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListBento(self, request, context):
        """Get a list of Bento that are stored in current repository
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ContainerizeBento(self, request, context):
        """Create a container image from a Bento
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UploadBento(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DownloadBento(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_YataiServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'HealthCheck': grpc.unary_unary_rpc_method_handler(
                    servicer.HealthCheck,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=yatai__service__pb2.HealthCheckResponse.SerializeToString,
            ),
            'GetYataiServiceVersion': grpc.unary_unary_rpc_method_handler(
                    servicer.GetYataiServiceVersion,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=yatai__service__pb2.GetYataiServiceVersionResponse.SerializeToString,
            ),
            'ApplyDeployment': grpc.unary_unary_rpc_method_handler(
                    servicer.ApplyDeployment,
                    request_deserializer=deployment__pb2.ApplyDeploymentRequest.FromString,
                    response_serializer=deployment__pb2.ApplyDeploymentResponse.SerializeToString,
            ),
            'DeleteDeployment': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteDeployment,
                    request_deserializer=deployment__pb2.DeleteDeploymentRequest.FromString,
                    response_serializer=deployment__pb2.DeleteDeploymentResponse.SerializeToString,
            ),
            'GetDeployment': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDeployment,
                    request_deserializer=deployment__pb2.GetDeploymentRequest.FromString,
                    response_serializer=deployment__pb2.GetDeploymentResponse.SerializeToString,
            ),
            'DescribeDeployment': grpc.unary_unary_rpc_method_handler(
                    servicer.DescribeDeployment,
                    request_deserializer=deployment__pb2.DescribeDeploymentRequest.FromString,
                    response_serializer=deployment__pb2.DescribeDeploymentResponse.SerializeToString,
            ),
            'ListDeployments': grpc.unary_unary_rpc_method_handler(
                    servicer.ListDeployments,
                    request_deserializer=deployment__pb2.ListDeploymentsRequest.FromString,
                    response_serializer=deployment__pb2.ListDeploymentsResponse.SerializeToString,
            ),
            'AddBento': grpc.unary_unary_rpc_method_handler(
                    servicer.AddBento,
                    request_deserializer=repository__pb2.AddBentoRequest.FromString,
                    response_serializer=repository__pb2.AddBentoResponse.SerializeToString,
            ),
            'UpdateBento': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateBento,
                    request_deserializer=repository__pb2.UpdateBentoRequest.FromString,
                    response_serializer=repository__pb2.UpdateBentoResponse.SerializeToString,
            ),
            'GetBento': grpc.unary_unary_rpc_method_handler(
                    servicer.GetBento,
                    request_deserializer=repository__pb2.GetBentoRequest.FromString,
                    response_serializer=repository__pb2.GetBentoResponse.SerializeToString,
            ),
            'DangerouslyDeleteBento': grpc.unary_unary_rpc_method_handler(
                    servicer.DangerouslyDeleteBento,
                    request_deserializer=repository__pb2.DangerouslyDeleteBentoRequest.FromString,
                    response_serializer=repository__pb2.DangerouslyDeleteBentoResponse.SerializeToString,
            ),
            'ListBento': grpc.unary_unary_rpc_method_handler(
                    servicer.ListBento,
                    request_deserializer=repository__pb2.ListBentoRequest.FromString,
                    response_serializer=repository__pb2.ListBentoResponse.SerializeToString,
            ),
            'ContainerizeBento': grpc.unary_unary_rpc_method_handler(
                    servicer.ContainerizeBento,
                    request_deserializer=repository__pb2.ContainerizeBentoRequest.FromString,
                    response_serializer=repository__pb2.ContainerizeBentoResponse.SerializeToString,
            ),
            'UploadBento': grpc.stream_unary_rpc_method_handler(
                    servicer.UploadBento,
                    request_deserializer=repository__pb2.UploadBentoRequest.FromString,
                    response_serializer=repository__pb2.UploadBentoResponse.SerializeToString,
            ),
            'DownloadBento': grpc.unary_stream_rpc_method_handler(
                    servicer.DownloadBento,
                    request_deserializer=repository__pb2.DownloadBentoRequest.FromString,
                    response_serializer=repository__pb2.DownloadBentoResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'bentoml.Yatai', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Yatai(object):
    """Yatai RPC Server

    A stateful service that provides a complete BentoML model management
    and model serving/deployment workflow

    It provides two sets of APIs:
    Bento Repository: Manages saved Bento files, and making them available
    for serving in production environments
    Serving Deployment: Deploys saved Bento to a varity of different cloud
    platforms, track deploym//ent status, set up logging
    monitoring for your model serving workloads, and
    """

    @staticmethod
    def HealthCheck(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/HealthCheck',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            yatai__service__pb2.HealthCheckResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetYataiServiceVersion(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/GetYataiServiceVersion',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            yatai__service__pb2.GetYataiServiceVersionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ApplyDeployment(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/ApplyDeployment',
            deployment__pb2.ApplyDeploymentRequest.SerializeToString,
            deployment__pb2.ApplyDeploymentResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteDeployment(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/DeleteDeployment',
            deployment__pb2.DeleteDeploymentRequest.SerializeToString,
            deployment__pb2.DeleteDeploymentResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDeployment(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/GetDeployment',
            deployment__pb2.GetDeploymentRequest.SerializeToString,
            deployment__pb2.GetDeploymentResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DescribeDeployment(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/DescribeDeployment',
            deployment__pb2.DescribeDeploymentRequest.SerializeToString,
            deployment__pb2.DescribeDeploymentResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListDeployments(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/ListDeployments',
            deployment__pb2.ListDeploymentsRequest.SerializeToString,
            deployment__pb2.ListDeploymentsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AddBento(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/AddBento',
            repository__pb2.AddBentoRequest.SerializeToString,
            repository__pb2.AddBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateBento(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/UpdateBento',
            repository__pb2.UpdateBentoRequest.SerializeToString,
            repository__pb2.UpdateBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetBento(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/GetBento',
            repository__pb2.GetBentoRequest.SerializeToString,
            repository__pb2.GetBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DangerouslyDeleteBento(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/DangerouslyDeleteBento',
            repository__pb2.DangerouslyDeleteBentoRequest.SerializeToString,
            repository__pb2.DangerouslyDeleteBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListBento(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/ListBento',
            repository__pb2.ListBentoRequest.SerializeToString,
            repository__pb2.ListBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ContainerizeBento(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bentoml.Yatai/ContainerizeBento',
            repository__pb2.ContainerizeBentoRequest.SerializeToString,
            repository__pb2.ContainerizeBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UploadBento(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/bentoml.Yatai/UploadBento',
            repository__pb2.UploadBentoRequest.SerializeToString,
            repository__pb2.UploadBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DownloadBento(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/bentoml.Yatai/DownloadBento',
            repository__pb2.DownloadBentoRequest.SerializeToString,
            repository__pb2.DownloadBentoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
