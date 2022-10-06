load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", "requirement")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_buf//buf:defs.bzl", "buf_lint_test")

# "public" targets are alias rules in //.
package(default_visibility = ["//:__subpackages__"])

filegroup(
    name = "srcs_files",
    srcs = glob(["*.proto"]) + glob(["*.py"]) + glob(["_generated_pb3/*.py"]) + glob(["_generated_pb4/*.py"]),
)

proto_library(
    name = "service_test_proto",
    srcs = ["service_test.proto"],
)

py_library(
    name = "proto_lib",
    srcs = glob(["_generated_pb3/*.py"]) + glob(["_generated_pb4/*.py"]) + glob(["*.py"]),
    data = ["//tests/proto:srcs_files"],
    imports = ["../../"],
    deps = [requirement("protobuf")],
)
