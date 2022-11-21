load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", "requirement")
load("@bazel_skylib//lib:selects.bzl", "selects")

package(default_visibility = ["//src/bentoml/_internal/frameworks:__pkg__"])

filegroup(
    name = "srcs_files",
    srcs = glob(["*.py"]),
)

py_library(
    name = "onnx",
    srcs = ["onnx.py"],
    deps = [
        "//src/bentoml",
        requirement("onnx"),
    ],
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    deps = selects.with_or({
        "@bazel_tools//src/conditions:darwin_arm64": [
            "//src/bentoml",
            "//:grpcio",
            requirement("tensorflow-macos"),
            requirement("importlib-metadata"),
        ],
        ("@bazel_tools//src/conditions:linux", "@bazel_tools//src/conditions:windows", "@bazel_tools//src/conditions:darwin_x86_64"): [
            "//src/bentoml",
            "//:grpcio",
            requirement("tensorflow"),
            requirement("importlib-metadata"),
        ],
    }),
)
