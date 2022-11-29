load("//rules/py:python.bzl", "py_test")

[
    py_test(name = lib)
    for lib in [
        "resource",
        "service_api",
        "store",
    ]
]

py_test(
    name = "model",
    srcs = ["models/test_model.py"],
)

filegroup(
    name = "conftest",
    srcs = ["configuration/conftest.py"],
    visibility = ["//:__subpackages__"],
)

py_test(
    name = "containers",
    srcs = ["configuration/test_containers.py"],
    data = [":conftest"],
)

py_test(
    name = "helpers",
    srcs = ["configuration/test_helpers.py"],
    data = [":conftest"],
)

py_test(
    name = "v1_migration",
    srcs = ["configuration/test_v1_migration.py"],
    data = [":conftest"],
)
