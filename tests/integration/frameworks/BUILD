load("//rules:python.bzl", "framework_py_test", _FRAMEWORKS = "FRAMEWORKS")
load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", pypi = "requirement")
load("@frameworks//:requirements.bzl", framework = "requirement")

package(default_visibility = ["//:__subpackages__"])

filegroup(
    name = "conftest",
    srcs = ["conftest.py"],
)

py_library(
    name = "models",
    srcs = ["//tests/integration/frameworks/models:__init__.py"],
    deps = [
        pypi("attrs"),
        pypi("numpy"),
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
    imports = [
        ".",
        "../",
    ],
    deps = [
        ":models",
        "//src/bentoml",
        pypi("pytest"),
        pypi("pytest-xdist"),
        pypi("pytest-asyncio"),
    ],
)

[
    framework_py_test(name = lib)
    for lib in _FRAMEWORKS
]

framework_py_test(name = "lightgbm")

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
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            framework("tensorflow"),
            framework("torch"),
            framework("flax"),
        ],
        "@bazel_tools//src/conditions:darwin": [
            framework("tensorflow-macos"),
            framework("torch"),
            framework("flax"),
        ],
    }),
)
