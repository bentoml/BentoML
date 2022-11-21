load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//tests:__subpackages__"])

py_library(
    name = "tests_lib",
    srcs = ["//tests/proto:srcs_files"] + glob(["proto/*.py"]),
    data = ["//tests/proto:srcs_files"],
    imports = ["../"],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            "@pypi_grpcio//:pkg",
            "@pypi_protobuf//:pkg",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//:grpcio",
            "//:protobuf",
        ],
    }),
)
