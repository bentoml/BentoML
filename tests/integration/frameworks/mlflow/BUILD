load("//rules:python.bzl", "py_test")
load("@frameworks//:requirements.bzl", framework = "requirement")

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
        framework("numpy"),
        framework("scikit-learn"),
        framework("mlflow"),
        framework("importlib-resources"),
    ],
)
