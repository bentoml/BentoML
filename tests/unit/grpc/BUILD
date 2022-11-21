load("//rules:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

filegroup(
    name = "conftest",
    srcs = ["conftest.py"],
)

py_test(
    name = "grpc_utils",
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            requirement("grpcio"),
            "//tests:tests_lib",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//:grpcio",
            "//tests:tests_lib",
        ],
    }),
)

py_test(
    name = "access",
    srcs = ["interceptors/test_access.py"],
    data = [
        ":conftest",
        "//tests/unit:conftest",
    ],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            requirement("grpcio"),
            requirement("protobuf"),
            "//tests:tests_lib",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//:grpcio",
            "//:protobuf",
            "//tests:tests_lib",
        ],
    }),
)

py_test(
    name = "prometheus",
    srcs = ["interceptors/test_prometheus.py"],
    data = [
        ":conftest",
        "//tests/unit:conftest",
    ],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            requirement("grpcio"),
            requirement("protobuf"),
            "//tests:tests_lib",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//:grpcio",
            "//:protobuf",
            "//tests:tests_lib",
        ],
    }),
)
