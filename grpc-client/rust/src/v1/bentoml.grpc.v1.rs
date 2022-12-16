/// ServiceMetadataRequest message doesn't take any arguments.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ServiceMetadataRequest {}
/// ServiceMetadataResponse returns metadata of bentoml.Service.
/// Currently it includes name, version, apis, and docs.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ServiceMetadataResponse {
    /// name is the service name.
    #[prost(string, tag = "1")]
    pub name: ::prost::alloc::string::String,
    /// apis holds a list of InferenceAPI of the service.
    #[prost(message, repeated, tag = "2")]
    pub apis: ::prost::alloc::vec::Vec<service_metadata_response::InferenceApi>,
    /// docs is the documentation of the service.
    #[prost(string, tag = "3")]
    pub docs: ::prost::alloc::string::String,
}
/// Nested message and enum types in `ServiceMetadataResponse`.
pub mod service_metadata_response {
    /// DescriptorMetadata is a metadata of any given IODescriptor.
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct DescriptorMetadata {
        /// descriptor_id describes the given ID of the descriptor, which matches with our OpenAPI definition.
        #[prost(string, optional, tag = "1")]
        pub descriptor_id: ::core::option::Option<::prost::alloc::string::String>,
        /// attributes is the kwargs of the given descriptor.
        #[prost(message, optional, tag = "2")]
        pub attributes: ::core::option::Option<::prost_types::Struct>,
    }
    /// InferenceAPI is bentoml._internal.service.inferece_api.InferenceAPI
    /// that is exposed to gRPC client.
    /// There is no way for reflection to get information of given @svc.api.
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceApi {
        /// name is the name of the API.
        #[prost(string, tag = "1")]
        pub name: ::prost::alloc::string::String,
        /// input is the input descriptor of the API.
        #[prost(message, optional, tag = "2")]
        pub input: ::core::option::Option<DescriptorMetadata>,
        /// output is the output descriptor of the API.
        #[prost(message, optional, tag = "3")]
        pub output: ::core::option::Option<DescriptorMetadata>,
        /// docs is the optional documentation of the API.
        #[prost(string, optional, tag = "4")]
        pub docs: ::core::option::Option<::prost::alloc::string::String>,
    }
}
/// Request message for incoming Call.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Request {
    /// api_name defines the API entrypoint to call.
    /// api_name is the name of the function defined in bentoml.Service.
    /// Example:
    ///
    ///      @svc.api(input=NumpyNdarray(), output=File())
    ///      def predict(input: NDArray\[float\]) -> bytes:
    ///          ...
    ///
    ///      api_name is "predict" in this case.
    #[prost(string, tag = "1")]
    pub api_name: ::prost::alloc::string::String,
    #[prost(oneof = "request::Content", tags = "3, 5, 6, 7, 8, 9, 10, 2")]
    pub content: ::core::option::Option<request::Content>,
}
/// Nested message and enum types in `Request`.
pub mod request {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Content {
        /// NDArray represents a n-dimensional array of arbitrary type.
        #[prost(message, tag = "3")]
        Ndarray(super::NdArray),
        /// DataFrame represents any tabular data type. We are using
        /// DataFrame as a trivial representation for tabular type.
        #[prost(message, tag = "5")]
        Dataframe(super::DataFrame),
        /// Series portrays a series of values. This can be used for
        /// representing Series types in tabular data.
        #[prost(message, tag = "6")]
        Series(super::Series),
        /// File represents for any arbitrary file type. This can be
        /// plaintext, image, video, audio, etc.
        #[prost(message, tag = "7")]
        File(super::File),
        /// Text represents a string inputs.
        #[prost(message, tag = "8")]
        Text(::prost::alloc::string::String),
        /// JSON is represented by using google.protobuf.Value.
        /// see <https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/struct.proto>
        #[prost(message, tag = "9")]
        Json(::prost_types::Value),
        /// Multipart represents a multipart message.
        /// It comprises of a mapping from given type name to a subset of aforementioned types.
        #[prost(message, tag = "10")]
        Multipart(super::Multipart),
        /// serialized_bytes is for data serialized in BentoML's internal serialization format.
        #[prost(bytes, tag = "2")]
        SerializedBytes(::prost::alloc::vec::Vec<u8>),
    }
}
/// Request message for incoming Call.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Response {
    #[prost(oneof = "response::Content", tags = "1, 3, 5, 6, 7, 8, 9, 2")]
    pub content: ::core::option::Option<response::Content>,
}
/// Nested message and enum types in `Response`.
pub mod response {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Content {
        /// NDArray represents a n-dimensional array of arbitrary type.
        #[prost(message, tag = "1")]
        Ndarray(super::NdArray),
        /// DataFrame represents any tabular data type. We are using
        /// DataFrame as a trivial representation for tabular type.
        #[prost(message, tag = "3")]
        Dataframe(super::DataFrame),
        /// Series portrays a series of values. This can be used for
        /// representing Series types in tabular data.
        #[prost(message, tag = "5")]
        Series(super::Series),
        /// File represents for any arbitrary file type. This can be
        /// plaintext, image, video, audio, etc.
        #[prost(message, tag = "6")]
        File(super::File),
        /// Text represents a string inputs.
        #[prost(message, tag = "7")]
        Text(::prost::alloc::string::String),
        /// JSON is represented by using google.protobuf.Value.
        /// see <https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/struct.proto>
        #[prost(message, tag = "8")]
        Json(::prost_types::Value),
        /// Multipart represents a multipart message.
        /// It comprises of a mapping from given type name to a subset of aforementioned types.
        #[prost(message, tag = "9")]
        Multipart(super::Multipart),
        /// serialized_bytes is for data serialized in BentoML's internal serialization format.
        #[prost(bytes, tag = "2")]
        SerializedBytes(::prost::alloc::vec::Vec<u8>),
    }
}
/// Part represents possible value types for multipart message.
/// These are the same as the types in Request message.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Part {
    #[prost(oneof = "part::Representation", tags = "1, 3, 5, 6, 7, 8, 4")]
    pub representation: ::core::option::Option<part::Representation>,
}
/// Nested message and enum types in `Part`.
pub mod part {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Representation {
        /// NDArray represents a n-dimensional array of arbitrary type.
        #[prost(message, tag = "1")]
        Ndarray(super::NdArray),
        /// DataFrame represents any tabular data type. We are using
        /// DataFrame as a trivial representation for tabular type.
        #[prost(message, tag = "3")]
        Dataframe(super::DataFrame),
        /// Series portrays a series of values. This can be used for
        /// representing Series types in tabular data.
        #[prost(message, tag = "5")]
        Series(super::Series),
        /// File represents for any arbitrary file type. This can be
        /// plaintext, image, video, audio, etc.
        #[prost(message, tag = "6")]
        File(super::File),
        /// Text represents a string inputs.
        #[prost(message, tag = "7")]
        Text(::prost::alloc::string::String),
        /// JSON is represented by using google.protobuf.Value.
        /// see <https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/struct.proto>
        #[prost(message, tag = "8")]
        Json(::prost_types::Value),
        /// serialized_bytes is for data serialized in BentoML's internal serialization format.
        #[prost(bytes, tag = "4")]
        SerializedBytes(::prost::alloc::vec::Vec<u8>),
    }
}
/// Multipart represents a multipart message.
/// It comprises of a mapping from given type name to a subset of aforementioned types.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Multipart {
    #[prost(map = "string, message", tag = "1")]
    pub fields: ::std::collections::HashMap<::prost::alloc::string::String, Part>,
}
/// File represents for any arbitrary file type. This can be
/// plaintext, image, video, audio, etc.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct File {
    /// optional file type, let it be csv, text, parquet, etc.
    /// v1alpha1 uses 1 as FileType enum.
    #[prost(string, optional, tag = "3")]
    pub kind: ::core::option::Option<::prost::alloc::string::String>,
    /// contents of file as bytes.
    #[prost(bytes = "vec", tag = "2")]
    pub content: ::prost::alloc::vec::Vec<u8>,
}
/// DataFrame represents any tabular data type. We are using
/// DataFrame as a trivial representation for tabular type.
/// This message carries given implementation of tabular data based on given orientation.
/// TODO: support index, records, etc.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DataFrame {
    /// columns name
    #[prost(string, repeated, tag = "1")]
    pub column_names: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// columns orient.
    /// { column ↠ { index ↠ value } }
    #[prost(message, repeated, tag = "2")]
    pub columns: ::prost::alloc::vec::Vec<Series>,
}
/// Series portrays a series of values. This can be used for
/// representing Series types in tabular data.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Series {
    /// A bool parameter value
    #[prost(bool, repeated, tag = "1")]
    pub bool_values: ::prost::alloc::vec::Vec<bool>,
    /// A float parameter value
    #[prost(float, repeated, tag = "2")]
    pub float_values: ::prost::alloc::vec::Vec<f32>,
    /// A int32 parameter value
    #[prost(int32, repeated, tag = "3")]
    pub int32_values: ::prost::alloc::vec::Vec<i32>,
    /// A int64 parameter value
    #[prost(int64, repeated, tag = "6")]
    pub int64_values: ::prost::alloc::vec::Vec<i64>,
    /// A string parameter value
    #[prost(string, repeated, tag = "5")]
    pub string_values: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// represents a double parameter value.
    #[prost(double, repeated, tag = "4")]
    pub double_values: ::prost::alloc::vec::Vec<f64>,
}
/// NDArray represents a n-dimensional array of arbitrary type.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NdArray {
    /// DTYPE is the data type of given array
    #[prost(enumeration = "nd_array::DType", tag = "1")]
    pub dtype: i32,
    /// shape is the shape of given array.
    #[prost(int32, repeated, tag = "2")]
    pub shape: ::prost::alloc::vec::Vec<i32>,
    /// represents a string parameter value.
    #[prost(string, repeated, tag = "5")]
    pub string_values: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// represents a float parameter value.
    #[prost(float, repeated, tag = "3")]
    pub float_values: ::prost::alloc::vec::Vec<f32>,
    /// represents a double parameter value.
    #[prost(double, repeated, tag = "4")]
    pub double_values: ::prost::alloc::vec::Vec<f64>,
    /// represents a bool parameter value.
    #[prost(bool, repeated, tag = "6")]
    pub bool_values: ::prost::alloc::vec::Vec<bool>,
    /// represents a int32 parameter value.
    #[prost(int32, repeated, tag = "7")]
    pub int32_values: ::prost::alloc::vec::Vec<i32>,
    /// represents a int64 parameter value.
    #[prost(int64, repeated, tag = "8")]
    pub int64_values: ::prost::alloc::vec::Vec<i64>,
    /// represents a uint32 parameter value.
    #[prost(uint32, repeated, tag = "9")]
    pub uint32_values: ::prost::alloc::vec::Vec<u32>,
    /// represents a uint64 parameter value.
    #[prost(uint64, repeated, tag = "10")]
    pub uint64_values: ::prost::alloc::vec::Vec<u64>,
}
/// Nested message and enum types in `NDArray`.
pub mod nd_array {
    /// Represents data type of a given array.
    #[derive(
        Clone,
        Copy,
        Debug,
        PartialEq,
        Eq,
        Hash,
        PartialOrd,
        Ord,
        ::prost::Enumeration
    )]
    #[repr(i32)]
    pub enum DType {
        /// Represents a None type.
        DtypeUnspecified = 0,
        /// Represents an float type.
        DtypeFloat = 1,
        /// Represents an double type.
        DtypeDouble = 2,
        /// Represents a bool type.
        DtypeBool = 3,
        /// Represents an int32 type.
        DtypeInt32 = 4,
        /// Represents an int64 type.
        DtypeInt64 = 5,
        /// Represents a uint32 type.
        DtypeUint32 = 6,
        /// Represents a uint64 type.
        DtypeUint64 = 7,
        /// Represents a string type.
        DtypeString = 8,
    }
    impl DType {
        /// String value of the enum field names used in the ProtoBuf definition.
        ///
        /// The values are not transformed in any way and thus are considered stable
        /// (if the ProtoBuf definition does not change) and safe for programmatic use.
        pub fn as_str_name(&self) -> &'static str {
            match self {
                DType::DtypeUnspecified => "DTYPE_UNSPECIFIED",
                DType::DtypeFloat => "DTYPE_FLOAT",
                DType::DtypeDouble => "DTYPE_DOUBLE",
                DType::DtypeBool => "DTYPE_BOOL",
                DType::DtypeInt32 => "DTYPE_INT32",
                DType::DtypeInt64 => "DTYPE_INT64",
                DType::DtypeUint32 => "DTYPE_UINT32",
                DType::DtypeUint64 => "DTYPE_UINT64",
                DType::DtypeString => "DTYPE_STRING",
            }
        }
    }
}
/// Generated client implementations.
pub mod bento_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    /// a gRPC BentoServer.
    #[derive(Debug, Clone)]
    pub struct BentoServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl BentoServiceClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> BentoServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> BentoServiceClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
            >>::Error: Into<StdError> + Send + Sync,
        {
            BentoServiceClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        /// Call handles methodcaller of given API entrypoint.
        pub async fn call(
            &mut self,
            request: impl tonic::IntoRequest<super::Request>,
        ) -> Result<tonic::Response<super::Response>, tonic::Status> {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/bentoml.grpc.v1.BentoService/Call",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ServiceMetadata returns metadata of bentoml.Service.
        pub async fn service_metadata(
            &mut self,
            request: impl tonic::IntoRequest<super::ServiceMetadataRequest>,
        ) -> Result<tonic::Response<super::ServiceMetadataResponse>, tonic::Status> {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/bentoml.grpc.v1.BentoService/ServiceMetadata",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod bento_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with BentoServiceServer.
    #[async_trait]
    pub trait BentoService: Send + Sync + 'static {
        /// Call handles methodcaller of given API entrypoint.
        async fn call(
            &self,
            request: tonic::Request<super::Request>,
        ) -> Result<tonic::Response<super::Response>, tonic::Status>;
        /// ServiceMetadata returns metadata of bentoml.Service.
        async fn service_metadata(
            &self,
            request: tonic::Request<super::ServiceMetadataRequest>,
        ) -> Result<tonic::Response<super::ServiceMetadataResponse>, tonic::Status>;
    }
    /// a gRPC BentoServer.
    #[derive(Debug)]
    pub struct BentoServiceServer<T: BentoService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: BentoService> BentoServiceServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            let inner = _Inner(inner);
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
            }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for BentoServiceServer<T>
    where
        T: BentoService,
        B: Body + Send + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            let inner = self.inner.clone();
            match req.uri().path() {
                "/bentoml.grpc.v1.BentoService/Call" => {
                    #[allow(non_camel_case_types)]
                    struct CallSvc<T: BentoService>(pub Arc<T>);
                    impl<T: BentoService> tonic::server::UnaryService<super::Request>
                    for CallSvc<T> {
                        type Response = super::Response;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::Request>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).call(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = CallSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/bentoml.grpc.v1.BentoService/ServiceMetadata" => {
                    #[allow(non_camel_case_types)]
                    struct ServiceMetadataSvc<T: BentoService>(pub Arc<T>);
                    impl<
                        T: BentoService,
                    > tonic::server::UnaryService<super::ServiceMetadataRequest>
                    for ServiceMetadataSvc<T> {
                        type Response = super::ServiceMetadataResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ServiceMetadataRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).service_metadata(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ServiceMetadataSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => {
                    Box::pin(async move {
                        Ok(
                            http::Response::builder()
                                .status(200)
                                .header("grpc-status", "12")
                                .header("content-type", "application/grpc")
                                .body(empty_body())
                                .unwrap(),
                        )
                    })
                }
            }
        }
    }
    impl<T: BentoService> Clone for BentoServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: BentoService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: BentoService> tonic::server::NamedService for BentoServiceServer<T> {
        const NAME: &'static str = "bentoml.grpc.v1.BentoService";
    }
}
