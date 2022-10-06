load("//rules/py:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

filegroup(
    name = "conftest",
    srcs = ["conftest.py"],
    visibility = ["//:__subpackages__"],
)

py_test(
    name = "grpc_utils",
    deps = [requirement("grpcio")],
)
