load("//rules/py:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

[
    py_test(name = lib)
    for lib in ("metrics", "uri")
]

py_test(
    name = "types",
    deps = [
        requirement("scipy"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)

py_test(
    name = "watchfilesplugin",
    srcs = ["circus/test_watchfilesplugin.py"],
    data = ["//tests/unit:conftest"],
    deps = [
        requirement("watchfiles"),
        requirement("psutil"),
    ],
)

py_test(
    name = "analytics",
    data = ["//tests/unit:conftest"],
)
