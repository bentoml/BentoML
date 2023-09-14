load("@rules_proto_grpc//go:defs.bzl", "go_grpc_library")
load("@io_bazel_rules_go//go:def.bzl", "go_binary")

proto_library(
    name = "service_v1_proto",
    srcs = ["bentoml/grpc/v1/service.proto"],
    deps = ["@com_google_protobuf//:struct_proto", "@com_google_protobuf//:wrappers_proto"],
)

go_grpc_library(
    name = "service_go",
    importpath = "github.com/bentoml/bentoml/grpc/v1",
    protos = [":service_v1_proto"],
)

go_binary(
    name = "client_go",
    srcs = ["client.go"],
    importpath = "github.com/bentoml/bentoml/grpc/v1",
    deps = [
        ":service_go",
        "@com_github_golang_protobuf//proto:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
        "@org_golang_google_grpc//credentials:go_default_library",
        "@org_golang_google_grpc//credentials/insecure:go_default_library",
    ],
)
