load("//rules:python.bzl", "py_test")
load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", "requirement")

filegroup(
    name = "config",
    srcs = glob(["configs/**"]) + [
        "bentofile.yaml",
        "nested/requirements.txt",
        "requirements.txt",
    ],
)

filegroup(
    name = "conftest",
    srcs = ["tests/conftest.py"],
)

filegroup(
    name = "srcs_files",
    srcs = glob(["*.py"]),
)

py_library(
    name = "pickle_model",
    srcs = ["pickle_model.py"],
    deps = [
        "//src/bentoml:picklable_model",
        requirement("pandas"),
        requirement("numpy"),
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":pickle_model",
        "//src/bentoml",
    ],
)

py_test(
    name = "e2e",
    srcs = ["tests/test_{}.py".format(lib) for lib in [
        "io",
        "meta",
    ]] + glob(["*.py"]),
    data = [
        ":config",
        ":conftest",
        ":srcs_files",
    ],
    deps = [
        ":train",
        requirement("numpy"),
        requirement("fastapi"),
    ],
)
