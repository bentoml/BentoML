load("//rules/py:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

package(default_visibility = ["//tests:__subpackages__"])

py_test(
    name = "cli_regression",
    deps = [requirement("click")],
)
