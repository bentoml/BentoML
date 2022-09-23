load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")

buildifier(
    name = "buildifier",
)

# Create a generated proto library
load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "service_proto",
    srcs = ["bentoml/grpc/v1alpha1/service.proto"],
    deps = [
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:wrappers_proto",
    ],
)

# compile cpp/client.cc
load("@rules_proto_grpc//cpp:defs.bzl", "cc_grpc_library", "cc_proto_library")

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
    srcs = ["cpp/client.cc"],
    deps = [
        ":service_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

# compile go/client.go
load("@rules_proto_grpc//go:defs.bzl", "go_grpc_library", "go_proto_library")
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_grpc_library(
    name = "service_go_grpc",
    importpath = "github.com/bentoml/bentoml/grpc/v1alpha1",
    protos = [":service_proto"],
)

go_library(
    name = "go_library",
    srcs = ["go/client.go"],
    importpath = "github.com/bentoml/bentoml/grpc/v1alpha1",
    deps = [
        ":service_go_grpc",
        "@com_github_golang_protobuf//proto:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
        "@org_golang_google_grpc//credentials:go_default_library",
        "@org_golang_google_grpc//credentials/insecure:go_default_library",
    ],
)

go_binary(
    name = "client_go",
    embed = [":go_library"],
)

# compile java/src/main/java/com/client/BentoServiceClient.java
load("@io_grpc_grpc_java//:java_grpc_library.bzl", "java_grpc_library")

java_proto_library(
    name = "service_java",
    deps = [":service_proto"],
)

java_grpc_library(
    name = "service_java_grpc",
    srcs = [":service_proto"],
    deps = [":service_java"],
)

java_library(
    name = "java_library",
    srcs = glob(["java/src/main/**/*.java"]),
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
