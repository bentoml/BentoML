load("//rules:python.bzl", "py_test")

[
    py_test(name = lib)
    for lib in [
        "metrics",
        "models",
    ]
]

filegroup(
    name = "conftest",
    srcs = ["conftest.py"],
    visibility = ["//:__subpackages__"],
)
