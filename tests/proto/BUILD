load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_buf//buf:defs.bzl", "buf_lint_test")

# "public" targets are alias rules in //.
package(default_visibility = ["//:__subpackages__"])

filegroup(
    name = "srcs_files",
    srcs = glob(["*.proto"]) + glob(["*.py"]),
)

proto_library(
    name = "service_test_proto",
    srcs = ["service_test.proto"],
)
