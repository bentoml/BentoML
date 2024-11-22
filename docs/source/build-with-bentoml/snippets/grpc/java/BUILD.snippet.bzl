load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "service_v1_proto",
    srcs = ["bentoml/grpc/v1/service.proto"],
    deps = [
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:wrappers_proto",
    ],
)

load("@io_grpc_grpc_java//:java_grpc_library.bzl", "java_grpc_library")

java_proto_library(
    name = "service_java",
    deps = [":service_v1_proto"],
)

java_grpc_library(
    name = "service_java_grpc",
    srcs = [":service_v1_proto"],
    deps = [":service_java"],
)

java_library(
    name = "java_library",
    srcs = glob(["client/java/src/main/**/*.java"]),
    runtime_deps = [
        "@io_grpc_grpc_java//netty",
    ],
    deps = [
        ":service_java",
        ":service_java_grpc",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
        "@io_grpc_grpc_java//api",
        "@io_grpc_grpc_java//protobuf",
        "@io_grpc_grpc_java//stub",
        "@maven//:com_google_api_grpc_proto_google_common_protos",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_code_gson_gson",
        "@maven//:com_google_guava_guava",
    ],
)

java_binary(
    name = "client_java",
    main_class = "com.client.BentoServiceClient",
    runtime_deps = [
        ":java_library",
    ],
)
