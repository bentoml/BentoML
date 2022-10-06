load("//rules/py:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

py_test(
    name = "access",
    data = [
        "//tests/unit:conftest",
        "//tests/unit/grpc:conftest",
    ],
    imports = ["../../../"],
    deps = [
        "//tests/proto:proto_lib",
        requirement("grpcio"),
    ],
)

py_test(
    name = "prometheus",
    data = [
        "//tests/unit:conftest",
        "//tests/unit/grpc:conftest",
    ],
    imports = ["../../../"],
    deps = [
        "//tests/proto:proto_lib",
        requirement("grpcio"),
        requirement("prometheus-client"),
    ],
)
