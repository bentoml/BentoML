load("@rules_python//python:defs.bzl", "py_library")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(":deps.bzl", "requirement")

package(default_visibility = ["//:__subpackges__"])

exports_files([
    "codecoverage.py",
    "deps.bzl",
])

bzl_library(
    name = "python.bzl",
    srcs = ["python.bzl"],
)

py_library(
    name = "codecov",
    testonly = True,
    srcs = ["codecoverage.py"],
    visibility = ["//visibility:public"],
    deps = [requirement("coverage")],
)
