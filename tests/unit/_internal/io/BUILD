load("//rules/py:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

[
    py_test(
        name = lib,
        deps = [
            "//:grpcio",
            "//:protobuf",
            requirement("numpy"),
            requirement("Pillow"),
            requirement("pydantic"),
        ],
    )
    for lib in [
        "base",
        "file",
        "image",
        "json",
        "multipart",
        "numpy",
        "text",
    ]
]
