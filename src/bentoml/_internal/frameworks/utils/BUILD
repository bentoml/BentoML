load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", pypi = "requirement")
load("@frameworks//:requirements.bzl", framework = "requirement")

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
        framework("onnx"),
    ],
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            "//src/bentoml",
            framework("grpcio"),
            framework("tensorflow"),
            pypi("importlib-metadata"),
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//src/bentoml",
            "//:grpcio",
            framework("tensorflow-macos"),
            pypi("importlib-metadata"),
        ],
    }),
)
