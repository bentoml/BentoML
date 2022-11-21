load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_buf//buf:defs.bzl", "buf_lint_test")

# "public" targets are alias rules in //.
package(default_visibility = ["//:__subpackages__"])

filegroup(
    name = "srcs_files",
    srcs = glob(["*.proto"]) + glob(["*.py"]),
)

proto_library(
    name = "service_proto",
    srcs = ["service.proto"],
    deps = [
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:wrappers_proto",
    ],
)

buf_lint_test(
    name = "service_proto_lint",
    config = "//src/bentoml/grpc:buf.yaml",
    targets = [":service_proto"],
)
