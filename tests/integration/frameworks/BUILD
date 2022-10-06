load("//rules/py:python.bzl", "framework_py_test", _FRAMEWORKS = "FRAMEWORKS")
load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", "requirement")
load("@bazel_skylib//lib:selects.bzl", "selects")

package(default_visibility = ["//:__subpackages__"])

filegroup(
    name = "conftest",
    srcs = ["conftest.py"],
)

py_library(
    name = "models",
    srcs = ["//tests/integration/frameworks/models:__init__.py"],
    imports = ["../../../"],
    deps = [
        requirement("attrs"),
        requirement("numpy"),
    ],
)

# NOTE: we need to create test_frameworks as a py_library such that
# we can then use it with models/*
py_library(
    name = "test_frameworks",
    srcs = ["test_frameworks.py"],
    data = [
        ":conftest",
        "//:pyproject",
    ],
    imports = ["."],
    deps = [
        ":models",
        "//src/bentoml",
        requirement("scikit-learn"),
        requirement("pytest"),
        requirement("pytest-xdist"),
        requirement("pytest-asyncio"),
    ],
)

[
    framework_py_test(name = lib)
    for lib in _FRAMEWORKS
]

framework_py_test(name = "lightgbm")

framework_py_test(
    name = "keras",
    deps = selects.with_or({
        "@bazel_tools//src/conditions:darwin_arm64": [
            requirement("grpcio"),
            requirement("tensorflow-macos"),
        ],
        ("@bazel_tools//src/conditions:linux", "@bazel_tools//src/conditions:windows", "@bazel_tools//src/conditions:darwin_x86_64"): [
            requirement("grpcio"),
            requirement("tensorflow"),
        ],
    }),
)

framework_py_test(
    name = "pytorch_lightning",
    deps = [
        requirement("grpcio"),
        requirement("torch"),
    ],
)

[
    framework_py_test(
        name = src,
        srcs = ["test_{src}_unit.py".format(src = src)],
    )
    for src in [
        "fastai",
        "pytorch",
        "tensorflow",
    ]
]

# transformers requires extra dependencies
framework_py_test(
    name = "transformers",
    srcs = [
        "test_transformers_unit.py",
        "//tests/integration/frameworks/models:transformers.py",
    ],
    deps = selects.with_or({
        "@bazel_tools//src/conditions:darwin_arm64": [
            requirement("grpcio"),
            requirement("tensorflow-macos"),
            requirement("torch"),
            requirement("flax"),
        ],
        "@bazel_tools//src/conditions:windows": [
            requirement("grpcio"),
            requirement("tensorflow"),
            requirement("torch"),
        ],
        ("@bazel_tools//src/conditions:linux", "@bazel_tools//src/conditions:darwin_x86_64"): [
            requirement("grpcio"),
            requirement("tensorflow"),
            requirement("torch"),
            requirement("flax"),
        ],
    }),
)
