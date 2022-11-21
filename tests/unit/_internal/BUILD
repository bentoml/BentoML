load("//rules/py:python.bzl", "py_test")

[
    py_test(name = lib)
    for lib in [
        "configuration",
        "resource",
        "service_api",
        "store",
    ]
]

py_test(
    name = "model",
    srcs = ["models/test_model.py"],
)
