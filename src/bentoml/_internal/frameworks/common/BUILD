load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", "requirement")

package(default_visibility = ["//src/bentoml/_internal/frameworks:__pkg__"])

filegroup(
    name = "srcs_files",
    srcs = glob(["*.py"]),
)

py_library(
    name = "pytorch",
    srcs = ["pytorch.py"],
    deps = [
        "//src/bentoml",
        requirement("simple-di"),
        requirement("torch"),
    ],
)
