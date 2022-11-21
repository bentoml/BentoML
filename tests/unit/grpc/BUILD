load("//rules/py:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

filegroup(
    name = "conftest",
    srcs = ["conftest.py"],
)

py_test(
    name = "grpc_utils",
    deps = [
        "//:grpcio",
        "//tests:tests_lib",
    ],
)

py_test(
    name = "access",
    srcs = ["interceptors/test_access.py"],
    data = [
        ":conftest",
        "//tests/unit:conftest",
    ],
    deps = [
        "//:grpcio",
        "//:protobuf",
        "//tests:tests_lib",
    ],
)

py_test(
    name = "prometheus",
    srcs = ["interceptors/test_prometheus.py"],
    data = [
        ":conftest",
        "//tests/unit:conftest",
    ],
    deps = [
        "//:grpcio",
        "//:protobuf",
        "//tests:tests_lib",
        requirement("prometheus-client"),
    ],
)
