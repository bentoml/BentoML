load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cc_grpc_library", "cc_proto_library")

proto_library(
    name = "service_v1_proto",
    srcs = ["bentoml/grpc/v1/service.proto"],
    deps = ["@com_google_protobuf//:struct_proto", "@com_google_protobuf//:wrappers_proto"],
)

cc_proto_library(
    name = "service_cc",
    protos = [":service_v1_proto"],
)

cc_grpc_library(
    name = "service_cc_grpc",
    protos = [":service_v1_proto"],
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
