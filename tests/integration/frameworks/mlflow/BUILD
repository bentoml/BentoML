load("//rules/py:python.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

filegroup(
    name = "testdata",
    srcs = glob(["MNIST/**"]) + glob(["NoPyfunc/**"]),
)

py_test(
    name = "apis",
    data = [
        ":testdata",
        "//tests/integration/frameworks:conftest",
    ],
    deps = [
        "//src/bentoml:mlflow",
        requirement("numpy"),
        requirement("scikit-learn"),
        requirement("mlflow"),
        requirement("importlib-resources"),
    ],
)
