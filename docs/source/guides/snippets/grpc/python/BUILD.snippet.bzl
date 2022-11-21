load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@pypi//:requirements.bzl", "requirement")

compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],
    requirements_in = "requirements.txt",
    requirements_txt = "requirements.lock.txt",
    visibility = ["//visibility:__pkg__"],
)

py_binary(
    name = "client",
    srcs = ["client.py"],
    main = "client.py",
    visibility = ["//visibility:public"],
    deps = [requirement("bentoml")],
)
