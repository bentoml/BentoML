"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import builtins
import collections.abc
import concurrent.futures
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import google.protobuf.service
import google.protobuf.struct_pb2
import google.protobuf.wrappers_pb2
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class ServiceMetadataRequest(google.protobuf.message.Message):
    """ServiceMetadataRequest message doesn't take any arguments."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___ServiceMetadataRequest = ServiceMetadataRequest

@typing_extensions.final
class ServiceMetadataResponse(google.protobuf.message.Message):
    """ServiceMetadataResponse returns metadata of bentoml.Service.
    Currently it includes name, version, apis, and docs.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class DescriptorMetadata(google.protobuf.message.Message):
        """DescriptorMetadata is a metadata of any given IODescriptor."""

        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        DESCRIPTOR_ID_FIELD_NUMBER: builtins.int
        ATTRIBUTES_FIELD_NUMBER: builtins.int
        descriptor_id: builtins.str
        """descriptor_id describes the given ID of the descriptor, which matches with our OpenAPI definition."""
        @property
        def attributes(self) -> google.protobuf.struct_pb2.Struct:
            """attributes is the kwargs of the given descriptor."""
        def __init__(
            self,
            *,
            descriptor_id: builtins.str | None = ...,
            attributes: google.protobuf.struct_pb2.Struct | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["_descriptor_id", b"_descriptor_id", "attributes", b"attributes", "descriptor_id", b"descriptor_id"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["_descriptor_id", b"_descriptor_id", "attributes", b"attributes", "descriptor_id", b"descriptor_id"]) -> None: ...
        def WhichOneof(self, oneof_group: typing_extensions.Literal["_descriptor_id", b"_descriptor_id"]) -> typing_extensions.Literal["descriptor_id"] | None: ...

    @typing_extensions.final
    class InferenceAPI(google.protobuf.message.Message):
        """InferenceAPI is bentoml._internal.service.inferece_api.InferenceAPI
        that is exposed to gRPC client.
        There is no way for reflection to get information of given @svc.api.
        """

        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        NAME_FIELD_NUMBER: builtins.int
        INPUT_FIELD_NUMBER: builtins.int
        OUTPUT_FIELD_NUMBER: builtins.int
        DOCS_FIELD_NUMBER: builtins.int
        name: builtins.str
        """name is the name of the API."""
        @property
        def input(self) -> global___ServiceMetadataResponse.DescriptorMetadata:
            """input is the input descriptor of the API."""
        @property
        def output(self) -> global___ServiceMetadataResponse.DescriptorMetadata:
            """output is the output descriptor of the API."""
        docs: builtins.str
        """docs is the optional documentation of the API."""
        def __init__(
            self,
            *,
            name: builtins.str = ...,
            input: global___ServiceMetadataResponse.DescriptorMetadata | None = ...,
            output: global___ServiceMetadataResponse.DescriptorMetadata | None = ...,
            docs: builtins.str | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["_docs", b"_docs", "_input", b"_input", "_output", b"_output", "docs", b"docs", "input", b"input", "output", b"output"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["_docs", b"_docs", "_input", b"_input", "_output", b"_output", "docs", b"docs", "input", b"input", "name", b"name", "output", b"output"]) -> None: ...
        @typing.overload
        def WhichOneof(self, oneof_group: typing_extensions.Literal["_docs", b"_docs"]) -> typing_extensions.Literal["docs"] | None: ...
        @typing.overload
        def WhichOneof(self, oneof_group: typing_extensions.Literal["_input", b"_input"]) -> typing_extensions.Literal["input"] | None: ...
        @typing.overload
        def WhichOneof(self, oneof_group: typing_extensions.Literal["_output", b"_output"]) -> typing_extensions.Literal["output"] | None: ...

    NAME_FIELD_NUMBER: builtins.int
    APIS_FIELD_NUMBER: builtins.int
    DOCS_FIELD_NUMBER: builtins.int
    name: builtins.str
    """name is the service name."""
    @property
    def apis(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ServiceMetadataResponse.InferenceAPI]:
        """apis holds a list of InferenceAPI of the service."""
    docs: builtins.str
    """docs is the documentation of the service."""
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        apis: collections.abc.Iterable[global___ServiceMetadataResponse.InferenceAPI] | None = ...,
        docs: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["apis", b"apis", "docs", b"docs", "name", b"name"]) -> None: ...

global___ServiceMetadataResponse = ServiceMetadataResponse

@typing_extensions.final
class Request(google.protobuf.message.Message):
    """Request message for incoming Call."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    API_NAME_FIELD_NUMBER: builtins.int
    NDARRAY_FIELD_NUMBER: builtins.int
    DATAFRAME_FIELD_NUMBER: builtins.int
    SERIES_FIELD_NUMBER: builtins.int
    FILE_FIELD_NUMBER: builtins.int
    TEXT_FIELD_NUMBER: builtins.int
    JSON_FIELD_NUMBER: builtins.int
    MULTIPART_FIELD_NUMBER: builtins.int
    SERIALIZED_BYTES_FIELD_NUMBER: builtins.int
    api_name: builtins.str
    """api_name defines the API entrypoint to call.
    api_name is the name of the function defined in bentoml.Service.
    Example:

        @svc.api(input=NumpyNdarray(), output=File())
        def predict(input: NDArray[float]) -> bytes:
            ...

        api_name is "predict" in this case.
    """
    @property
    def ndarray(self) -> global___NDArray:
        """NDArray represents a n-dimensional array of arbitrary type."""
    @property
    def dataframe(self) -> global___DataFrame:
        """DataFrame represents any tabular data type. We are using
        DataFrame as a trivial representation for tabular type.
        """
    @property
    def series(self) -> global___Series:
        """Series portrays a series of values. This can be used for
        representing Series types in tabular data.
        """
    @property
    def file(self) -> global___File:
        """File represents for any arbitrary file type. This can be
        plaintext, image, video, audio, etc.
        """
    @property
    def text(self) -> google.protobuf.wrappers_pb2.StringValue:
        """Text represents a string inputs."""
    @property
    def json(self) -> google.protobuf.struct_pb2.Value:
        """JSON is represented by using google.protobuf.Value.
        see https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/struct.proto
        """
    @property
    def multipart(self) -> global___Multipart:
        """Multipart represents a multipart message.
        It comprises of a mapping from given type name to a subset of aforementioned types.
        """
    serialized_bytes: builtins.bytes
    """serialized_bytes is for data serialized in BentoML's internal serialization format."""
    def __init__(
        self,
        *,
        api_name: builtins.str = ...,
        ndarray: global___NDArray | None = ...,
        dataframe: global___DataFrame | None = ...,
        series: global___Series | None = ...,
        file: global___File | None = ...,
        text: google.protobuf.wrappers_pb2.StringValue | None = ...,
        json: google.protobuf.struct_pb2.Value | None = ...,
        multipart: global___Multipart | None = ...,
        serialized_bytes: builtins.bytes = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["content", b"content", "dataframe", b"dataframe", "file", b"file", "json", b"json", "multipart", b"multipart", "ndarray", b"ndarray", "serialized_bytes", b"serialized_bytes", "series", b"series", "text", b"text"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["api_name", b"api_name", "content", b"content", "dataframe", b"dataframe", "file", b"file", "json", b"json", "multipart", b"multipart", "ndarray", b"ndarray", "serialized_bytes", b"serialized_bytes", "series", b"series", "text", b"text"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["content", b"content"]) -> typing_extensions.Literal["ndarray", "dataframe", "series", "file", "text", "json", "multipart", "serialized_bytes"] | None: ...

global___Request = Request

@typing_extensions.final
class Response(google.protobuf.message.Message):
    """Request message for incoming Call."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NDARRAY_FIELD_NUMBER: builtins.int
    DATAFRAME_FIELD_NUMBER: builtins.int
    SERIES_FIELD_NUMBER: builtins.int
    FILE_FIELD_NUMBER: builtins.int
    TEXT_FIELD_NUMBER: builtins.int
    JSON_FIELD_NUMBER: builtins.int
    MULTIPART_FIELD_NUMBER: builtins.int
    SERIALIZED_BYTES_FIELD_NUMBER: builtins.int
    @property
    def ndarray(self) -> global___NDArray:
        """NDArray represents a n-dimensional array of arbitrary type."""
    @property
    def dataframe(self) -> global___DataFrame:
        """DataFrame represents any tabular data type. We are using
        DataFrame as a trivial representation for tabular type.
        """
    @property
    def series(self) -> global___Series:
        """Series portrays a series of values. This can be used for
        representing Series types in tabular data.
        """
    @property
    def file(self) -> global___File:
        """File represents for any arbitrary file type. This can be
        plaintext, image, video, audio, etc.
        """
    @property
    def text(self) -> google.protobuf.wrappers_pb2.StringValue:
        """Text represents a string inputs."""
    @property
    def json(self) -> google.protobuf.struct_pb2.Value:
        """JSON is represented by using google.protobuf.Value.
        see https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/struct.proto
        """
    @property
    def multipart(self) -> global___Multipart:
        """Multipart represents a multipart message.
        It comprises of a mapping from given type name to a subset of aforementioned types.
        """
    serialized_bytes: builtins.bytes
    """serialized_bytes is for data serialized in BentoML's internal serialization format."""
    def __init__(
        self,
        *,
        ndarray: global___NDArray | None = ...,
        dataframe: global___DataFrame | None = ...,
        series: global___Series | None = ...,
        file: global___File | None = ...,
        text: google.protobuf.wrappers_pb2.StringValue | None = ...,
        json: google.protobuf.struct_pb2.Value | None = ...,
        multipart: global___Multipart | None = ...,
        serialized_bytes: builtins.bytes = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["content", b"content", "dataframe", b"dataframe", "file", b"file", "json", b"json", "multipart", b"multipart", "ndarray", b"ndarray", "serialized_bytes", b"serialized_bytes", "series", b"series", "text", b"text"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["content", b"content", "dataframe", b"dataframe", "file", b"file", "json", b"json", "multipart", b"multipart", "ndarray", b"ndarray", "serialized_bytes", b"serialized_bytes", "series", b"series", "text", b"text"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["content", b"content"]) -> typing_extensions.Literal["ndarray", "dataframe", "series", "file", "text", "json", "multipart", "serialized_bytes"] | None: ...

global___Response = Response

@typing_extensions.final
class Part(google.protobuf.message.Message):
    """Part represents possible value types for multipart message.
    These are the same as the types in Request message.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NDARRAY_FIELD_NUMBER: builtins.int
    DATAFRAME_FIELD_NUMBER: builtins.int
    SERIES_FIELD_NUMBER: builtins.int
    FILE_FIELD_NUMBER: builtins.int
    TEXT_FIELD_NUMBER: builtins.int
    JSON_FIELD_NUMBER: builtins.int
    SERIALIZED_BYTES_FIELD_NUMBER: builtins.int
    @property
    def ndarray(self) -> global___NDArray:
        """NDArray represents a n-dimensional array of arbitrary type."""
    @property
    def dataframe(self) -> global___DataFrame:
        """DataFrame represents any tabular data type. We are using
        DataFrame as a trivial representation for tabular type.
        """
    @property
    def series(self) -> global___Series:
        """Series portrays a series of values. This can be used for
        representing Series types in tabular data.
        """
    @property
    def file(self) -> global___File:
        """File represents for any arbitrary file type. This can be
        plaintext, image, video, audio, etc.
        """
    @property
    def text(self) -> google.protobuf.wrappers_pb2.StringValue:
        """Text represents a string inputs."""
    @property
    def json(self) -> google.protobuf.struct_pb2.Value:
        """JSON is represented by using google.protobuf.Value.
        see https://github.com/protocolbuffers/protobuf/blob/main/src/google/protobuf/struct.proto
        """
    serialized_bytes: builtins.bytes
    """serialized_bytes is for data serialized in BentoML's internal serialization format."""
    def __init__(
        self,
        *,
        ndarray: global___NDArray | None = ...,
        dataframe: global___DataFrame | None = ...,
        series: global___Series | None = ...,
        file: global___File | None = ...,
        text: google.protobuf.wrappers_pb2.StringValue | None = ...,
        json: google.protobuf.struct_pb2.Value | None = ...,
        serialized_bytes: builtins.bytes = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["dataframe", b"dataframe", "file", b"file", "json", b"json", "ndarray", b"ndarray", "representation", b"representation", "serialized_bytes", b"serialized_bytes", "series", b"series", "text", b"text"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["dataframe", b"dataframe", "file", b"file", "json", b"json", "ndarray", b"ndarray", "representation", b"representation", "serialized_bytes", b"serialized_bytes", "series", b"series", "text", b"text"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["representation", b"representation"]) -> typing_extensions.Literal["ndarray", "dataframe", "series", "file", "text", "json", "serialized_bytes"] | None: ...

global___Part = Part

@typing_extensions.final
class Multipart(google.protobuf.message.Message):
    """Multipart represents a multipart message.
    It comprises of a mapping from given type name to a subset of aforementioned types.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class FieldsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        @property
        def value(self) -> global___Part: ...
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: global___Part | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    FIELDS_FIELD_NUMBER: builtins.int
    @property
    def fields(self) -> google.protobuf.internal.containers.MessageMap[builtins.str, global___Part]: ...
    def __init__(
        self,
        *,
        fields: collections.abc.Mapping[builtins.str, global___Part] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["fields", b"fields"]) -> None: ...

global___Multipart = Multipart

@typing_extensions.final
class File(google.protobuf.message.Message):
    """File represents for any arbitrary file type. This can be
    plaintext, image, video, audio, etc.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    KIND_FIELD_NUMBER: builtins.int
    CONTENT_FIELD_NUMBER: builtins.int
    kind: builtins.str
    """optional file type, let it be csv, text, parquet, etc.
    v1alpha1 uses 1 as FileType enum.
    """
    content: builtins.bytes
    """contents of file as bytes."""
    def __init__(
        self,
        *,
        kind: builtins.str | None = ...,
        content: builtins.bytes = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_kind", b"_kind", "kind", b"kind"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_kind", b"_kind", "content", b"content", "kind", b"kind"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_kind", b"_kind"]) -> typing_extensions.Literal["kind"] | None: ...

global___File = File

@typing_extensions.final
class DataFrame(google.protobuf.message.Message):
    """DataFrame represents any tabular data type. We are using
    DataFrame as a trivial representation for tabular type.
    This message carries given implementation of tabular data based on given orientation.
    TODO: support index, records, etc.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    COLUMN_NAMES_FIELD_NUMBER: builtins.int
    COLUMNS_FIELD_NUMBER: builtins.int
    @property
    def column_names(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """columns name"""
    @property
    def columns(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Series]:
        """columns orient.
        { column ↠ { index ↠ value } }
        """
    def __init__(
        self,
        *,
        column_names: collections.abc.Iterable[builtins.str] | None = ...,
        columns: collections.abc.Iterable[global___Series] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["column_names", b"column_names", "columns", b"columns"]) -> None: ...

global___DataFrame = DataFrame

@typing_extensions.final
class Series(google.protobuf.message.Message):
    """Series portrays a series of values. This can be used for
    representing Series types in tabular data.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BOOL_VALUES_FIELD_NUMBER: builtins.int
    FLOAT_VALUES_FIELD_NUMBER: builtins.int
    INT32_VALUES_FIELD_NUMBER: builtins.int
    INT64_VALUES_FIELD_NUMBER: builtins.int
    STRING_VALUES_FIELD_NUMBER: builtins.int
    DOUBLE_VALUES_FIELD_NUMBER: builtins.int
    @property
    def bool_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bool]:
        """A bool parameter value"""
    @property
    def float_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """A float parameter value"""
    @property
    def int32_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """A int32 parameter value"""
    @property
    def int64_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """A int64 parameter value"""
    @property
    def string_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """A string parameter value"""
    @property
    def double_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """represents a double parameter value."""
    def __init__(
        self,
        *,
        bool_values: collections.abc.Iterable[builtins.bool] | None = ...,
        float_values: collections.abc.Iterable[builtins.float] | None = ...,
        int32_values: collections.abc.Iterable[builtins.int] | None = ...,
        int64_values: collections.abc.Iterable[builtins.int] | None = ...,
        string_values: collections.abc.Iterable[builtins.str] | None = ...,
        double_values: collections.abc.Iterable[builtins.float] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["bool_values", b"bool_values", "double_values", b"double_values", "float_values", b"float_values", "int32_values", b"int32_values", "int64_values", b"int64_values", "string_values", b"string_values"]) -> None: ...

global___Series = Series

@typing_extensions.final
class NDArray(google.protobuf.message.Message):
    """NDArray represents a n-dimensional array of arbitrary type."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _DType:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _DTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[NDArray._DType.ValueType], builtins.type):  # noqa: F821
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        DTYPE_UNSPECIFIED: NDArray._DType.ValueType  # 0
        """Represents a None type."""
        DTYPE_FLOAT: NDArray._DType.ValueType  # 1
        """Represents an float type."""
        DTYPE_DOUBLE: NDArray._DType.ValueType  # 2
        """Represents an double type."""
        DTYPE_BOOL: NDArray._DType.ValueType  # 3
        """Represents a bool type."""
        DTYPE_INT32: NDArray._DType.ValueType  # 4
        """Represents an int32 type."""
        DTYPE_INT64: NDArray._DType.ValueType  # 5
        """Represents an int64 type."""
        DTYPE_UINT32: NDArray._DType.ValueType  # 6
        """Represents a uint32 type."""
        DTYPE_UINT64: NDArray._DType.ValueType  # 7
        """Represents a uint64 type."""
        DTYPE_STRING: NDArray._DType.ValueType  # 8
        """Represents a string type."""

    class DType(_DType, metaclass=_DTypeEnumTypeWrapper):
        """Represents data type of a given array."""

    DTYPE_UNSPECIFIED: NDArray.DType.ValueType  # 0
    """Represents a None type."""
    DTYPE_FLOAT: NDArray.DType.ValueType  # 1
    """Represents an float type."""
    DTYPE_DOUBLE: NDArray.DType.ValueType  # 2
    """Represents an double type."""
    DTYPE_BOOL: NDArray.DType.ValueType  # 3
    """Represents a bool type."""
    DTYPE_INT32: NDArray.DType.ValueType  # 4
    """Represents an int32 type."""
    DTYPE_INT64: NDArray.DType.ValueType  # 5
    """Represents an int64 type."""
    DTYPE_UINT32: NDArray.DType.ValueType  # 6
    """Represents a uint32 type."""
    DTYPE_UINT64: NDArray.DType.ValueType  # 7
    """Represents a uint64 type."""
    DTYPE_STRING: NDArray.DType.ValueType  # 8
    """Represents a string type."""

    DTYPE_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    STRING_VALUES_FIELD_NUMBER: builtins.int
    FLOAT_VALUES_FIELD_NUMBER: builtins.int
    DOUBLE_VALUES_FIELD_NUMBER: builtins.int
    BOOL_VALUES_FIELD_NUMBER: builtins.int
    INT32_VALUES_FIELD_NUMBER: builtins.int
    INT64_VALUES_FIELD_NUMBER: builtins.int
    UINT32_VALUES_FIELD_NUMBER: builtins.int
    UINT64_VALUES_FIELD_NUMBER: builtins.int
    dtype: global___NDArray.DType.ValueType
    """DTYPE is the data type of given array"""
    @property
    def shape(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """shape is the shape of given array."""
    @property
    def string_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """represents a string parameter value."""
    @property
    def float_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """represents a float parameter value."""
    @property
    def double_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """represents a double parameter value."""
    @property
    def bool_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bool]:
        """represents a bool parameter value."""
    @property
    def int32_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """represents a int32 parameter value."""
    @property
    def int64_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """represents a int64 parameter value."""
    @property
    def uint32_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """represents a uint32 parameter value."""
    @property
    def uint64_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """represents a uint64 parameter value."""
    def __init__(
        self,
        *,
        dtype: global___NDArray.DType.ValueType = ...,
        shape: collections.abc.Iterable[builtins.int] | None = ...,
        string_values: collections.abc.Iterable[builtins.str] | None = ...,
        float_values: collections.abc.Iterable[builtins.float] | None = ...,
        double_values: collections.abc.Iterable[builtins.float] | None = ...,
        bool_values: collections.abc.Iterable[builtins.bool] | None = ...,
        int32_values: collections.abc.Iterable[builtins.int] | None = ...,
        int64_values: collections.abc.Iterable[builtins.int] | None = ...,
        uint32_values: collections.abc.Iterable[builtins.int] | None = ...,
        uint64_values: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["bool_values", b"bool_values", "double_values", b"double_values", "dtype", b"dtype", "float_values", b"float_values", "int32_values", b"int32_values", "int64_values", b"int64_values", "shape", b"shape", "string_values", b"string_values", "uint32_values", b"uint32_values", "uint64_values", b"uint64_values"]) -> None: ...

global___NDArray = NDArray

class BentoService(google.protobuf.service.Service, metaclass=abc.ABCMeta):
    """a gRPC BentoServer."""

    DESCRIPTOR: google.protobuf.descriptor.ServiceDescriptor
    @abc.abstractmethod
    def Call(
        inst: BentoService,
        rpc_controller: google.protobuf.service.RpcController,
        request: global___Request,
        callback: collections.abc.Callable[[global___Response], None] | None,
    ) -> concurrent.futures.Future[global___Response]:
        """Call handles methodcaller of given API entrypoint."""
    @abc.abstractmethod
    def ServiceMetadata(
        inst: BentoService,
        rpc_controller: google.protobuf.service.RpcController,
        request: global___ServiceMetadataRequest,
        callback: collections.abc.Callable[[global___ServiceMetadataResponse], None] | None,
    ) -> concurrent.futures.Future[global___ServiceMetadataResponse]:
        """ServiceMetadata returns metadata of bentoml.Service."""

class BentoService_Stub(BentoService):
    """a gRPC BentoServer."""

    def __init__(self, rpc_channel: google.protobuf.service.RpcChannel) -> None: ...
    DESCRIPTOR: google.protobuf.descriptor.ServiceDescriptor
    def Call(
        inst: BentoService_Stub,
        rpc_controller: google.protobuf.service.RpcController,
        request: global___Request,
        callback: collections.abc.Callable[[global___Response], None] | None = ...,
    ) -> concurrent.futures.Future[global___Response]:
        """Call handles methodcaller of given API entrypoint."""
    def ServiceMetadata(
        inst: BentoService_Stub,
        rpc_controller: google.protobuf.service.RpcController,
        request: global___ServiceMetadataRequest,
        callback: collections.abc.Callable[[global___ServiceMetadataResponse], None] | None = ...,
    ) -> concurrent.futures.Future[global___ServiceMetadataResponse]:
        """ServiceMetadata returns metadata of bentoml.Service."""
