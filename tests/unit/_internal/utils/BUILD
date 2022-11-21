load("//rules:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

py_test(name = "metrics")

py_test(name = "uri")

py_test(name = "types")

py_test(
    name = "watchfilesplugin",
    srcs = ["circus/test_watchfilesplugin.py"],
    data = ["//tests/unit:conftest"],
    deps = [requirement("watchfiles")],
)

py_test(
    name = "analytics",
    data = ["//tests/unit:conftest"],
)
